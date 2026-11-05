# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from aklr.core.contract.locate import schema_op_dir
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import WorkloadSpec, axes_text
from aklr.core.platform.env import RuntimeEnv
from aklr.core.validation.stages import _measure
from aklr.core.validation.stages._bench._evaluation import (
    CandidateResult,
    ReferenceWorkloadResult,
    TrialRequest,
    TrialResult,
    run_trial,
)
from aklr.core.validation.stages._bench._execution import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ExecutionError,
    Executor,
)
from aklr.core.validation.stages._bench._report import BenchReport, WorkloadRecord
from aklr.core.validation.stages._measure import Threshold, Versus
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages.stage import PipelineStage
from aklr.core.validation.stages.validation_scope import ValidationScope

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport

__all__ = [
    "BUILD_TIMEOUT_SECONDS", "MAX_BATCH_OUTPUT_BYTES", "MAX_BATCH_WORKLOADS",
    "EVALUATION_SECONDS_PER_OUTPUT_GIB", "WORKLOAD_TIMEOUT_SECONDS", "BenchStage",
]  # fmt: skip

MAX_BATCH_OUTPUT_BYTES: Final = 16 << 30
# one batch runs on one device: a long schema split up spreads over all of them
MAX_BATCH_WORKLOADS: Final = 16
BUILD_TIMEOUT_SECONDS: Final = 900.0
WORKLOAD_TIMEOUT_SECONDS: Final = 300.0
# a trial's evaluation (references and compares) grows with the outputs: at m=342073, gemm_n3072_k512's
# 23 GiB took about 420 s on a busy MI355X node compared in float64 on the CPU, far less on the GPU
EVALUATION_SECONDS_PER_OUTPUT_GIB: Final = 60.0


def _evaluation_timeout_seconds(requests: Sequence[TrialRequest]) -> float:
    output_gib = sum(r.output_bytes for r in requests) / (1 << 30)
    # every rank's outputs, each against its own reference
    seconds = WORKLOAD_TIMEOUT_SECONDS * len(requests) + EVALUATION_SECONDS_PER_OUTPUT_GIB * output_gib
    return seconds * requests[0].n_ranks


class UnreadableIncumbent(RuntimeError):
    """A base schema of the same name is not even JSON: its replacement is never timed
    without it."""


class BenchStage(PipelineStage[BenchReport]):
    """Each schema's workloads in batches, a trial task per batch and device (``run_trial``):
    the candidate in its child task, then the reference."""

    name = "bench"
    report_type = BenchReport

    def __init__(self, devices: Sequence[str]) -> None:
        if not devices:
            raise ValueError("no device to run on")
        self._devices = tuple(devices)

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> BenchReport:
        if not scope.bench_paths:
            return BenchReport.skipped("no kernel to measure")
        self.progress.write(f"reading {len(scope.bench_paths)} schemas")
        registry = ContractRegistry.default()
        unreadable: list[Finding] = []
        batches: list[tuple[TrialRequest, ...]] = []
        for path in scope.bench_paths:
            try:
                batches += self._workloads_to_run(scope, path, registry)
            except UnreadableIncumbent as e:
                unreadable.append(
                    Finding(
                        code=FindingCode.PLATFORM_FAILURE,
                        schema_path=scope.relative_path(path),
                        axes=None,
                        detail=str(e),
                    )
                )
        self.progress.write(
            f"{len(batches)} batches ({len(scope.bench_paths)} schemas),"
            f" {sum(map(len, batches))} workloads, {len(self._devices)} devices"
        )
        done = itertools.count(1)
        # the costliest batches start first, so none of them is left as the tail:
        # a workload's checks and timing cost about one unit, and each GiB of its
        # inputs, drawn and copied for every step, about one more
        order = sorted(
            range(len(batches)),
            key=lambda i: -sum(1 + r.input_bytes / 2**30 for r in batches[i]),
        )
        by_ranks: dict[int, list[int]] = {}
        for i in order:
            by_ranks.setdefault(batches[i][0].n_ranks, []).append(i)
        by_index: dict[int, tuple[list[Finding], list[WorkloadRecord]]] = {}
        # one phase per group size, the largest first: a group's devices are leased together
        for n_ranks, indices in sorted(by_ranks.items(), reverse=True):
            if n_ranks > len(self._devices):
                detail = f"{n_ranks} ranks call the op together, {len(self._devices)} devices"
                for i in indices:
                    failure = Finding.unlocated(FindingCode.NO_MATCHING_DEVICE, detail)
                    by_index[i] = self._judged_findings(scope, batches[i], [failure], None, None)
                    self.progress.write(f"{next(done)}/{len(batches)} {batches[i][0].schema_path}: {detail}")
                continue
            with (
                Executor.for_devices(self._devices, devices_per_task=n_ranks) as executor,
                ThreadPoolExecutor(len(self._devices) // n_ranks) as pool,
            ):
                run = lambda i: self._run_batch(scope, executor, batches[i], done, len(batches))  # noqa: B023
                by_index |= zip(indices, pool.map(run, indices), strict=True)
        outcomes = [by_index[i] for i in range(len(batches))]
        findings = [*unreadable, *(f for found, _ in outcomes for f in found)]
        records = [r for _, batch_records in outcomes for r in batch_records]
        return BenchReport.from_findings(findings, workload_records=tuple(records))

    def _workloads_to_run(
        self,
        scope: ValidationScope,
        path: Path,
        registry: ContractRegistry,
    ) -> list[tuple[TrialRequest, ...]]:
        base_sha = scope.base_sha
        path = path.resolve()
        schema_path = path.relative_to(scope.root.resolve()).as_posix()
        schema = registry.parse_schema(path.read_bytes(), schema_path)
        if not schema.kernel.sources:  # a definition: no kernel to bench
            return []
        contract = registry.get(schema.contract_id)
        baseline = contract.baseline()
        defined_baseline = None if baseline is None else f"{baseline.__module__}.{baseline.__qualname__}"
        # dispatch's rule: a kernel must beat the defined baseline, if there is one
        min_speedup = RuntimeEnv().min_speedup if defined_baseline is not None else 0
        incumbents = self._incumbents(scope, schema_path, schema, contract, min_speedup) if base_sha else {}
        # A first release has no incumbent; per-kernel admission is in BenchReport.
        first_release = scope.is_first_release
        # the whole json the semantics read (the file holds the values at the Point)
        schema_json = schema.model_dump(mode="json")
        workloads_json = {axes_text(w["axes"]): w for w in schema_json["workloads"]}
        content_hash = schema.kernel.content_hash
        batches: list[tuple[TrialRequest, ...]] = []
        batch: list[TrialRequest] = []
        batch_bytes = 0
        # even batches: a short remainder would start late among other schemas'
        n_batches = max(1, -(-len(schema.workloads) // MAX_BATCH_WORKLOADS))
        sizes = [
            len(schema.workloads) // n_batches + (k < len(schema.workloads) % n_batches)
            for k in range(n_batches)
        ]
        for workload in schema.workloads:
            workload_axes_text = axes_text(workload.axes)
            point = schema.point_with_contract_id(workload.axes, contract.contract_id)
            contract.check_point(point)
            incumbent, incumbent_path, is_served = incumbents.get(workload_axes_text, (None, None, False))
            is_kept = (
                not base_sha
                or first_release
                or (incumbent is not None and incumbent.kernel.same_kernel(schema.kernel))
            )
            thresholds: dict[Versus, Threshold] = {"baseline": "retain" if is_kept else "entry"}
            if incumbent is not None and incumbent.kernel.content_hash != content_hash:
                thresholds["incumbent"] = "retain"
            else:
                incumbent = None  # this very kernel: never timed against itself
            if defined_baseline is None:
                # the incumbent times as the baseline; with none, nothing is timed
                thresholds = {"baseline": "retain"} if incumbent is not None else {}
            request = TrialRequest(
                schema_path=str(path),
                schema=schema,
                contract=contract,
                workload=workload,
                point=point,
                axes_text=workload_axes_text,
                schema_json=schema_json,
                workload_json=workloads_json[workload_axes_text],
                distributions=schema.distributions_at(point),
                thresholds=thresholds,
                min_speedup=min_speedup,
                incumbent=incumbent,
                # what users run today: the served base kernel, this one included
                baseline=(incumbent_path if is_served else None) or defined_baseline,
            )
            n_bytes = request.output_bytes
            if batch and (
                batch_bytes + n_bytes > MAX_BATCH_OUTPUT_BYTES
                or len(batch) == sizes[min(len(batches), n_batches - 1)]
            ):
                batches.append(tuple(batch))
                batch, batch_bytes = [], 0
            batch.append(request)
            batch_bytes += n_bytes
        if batch:
            batches.append(tuple(batch))
        return batches

    def _incumbents(
        self,
        scope: ValidationScope,
        schema_path: str,
        schema: OpContract,
        contract: OpContract,
        min_speedup: float,
    ) -> dict[str, tuple[OpContract, str, bool]]:
        """Per workload, the base kernel dispatch serves at its Point (or this very
        kernel at base), its path, and whether dispatch serves it."""
        impls_dir = (schema_op_dir(Path(schema_path)) / "impls").as_posix()
        paths = [
            p
            for p in scope.paths_at_base(impls_dir)
            if ValidationScope.path_kind(p) == "schema_data" and PurePosixPath(p).parent.name == schema.name
        ]
        served: dict[Any, list[tuple[OpContract, WorkloadSpec, str]]] = {}
        for path in paths:
            text = scope.text_at_base(path)
            try:
                base = ContractRegistry.default().parse_schema(text.encode(), path)
            except ValueError as e:
                try:  # JSON, so base CI read it: a schema of a contract this change rewrote
                    json.loads(text)
                except ValueError:
                    raise UnreadableIncumbent(f"cannot read the incumbent {path}: {e}") from e
                continue
            entries = []
            for workload in base.workloads:
                try:  # a Point of other keys: another contract version's base
                    point = base.point_with_contract_id(workload.axes, contract.contract_id)
                    contract.check_point(point)
                except ValueError:
                    entries = []
                    break
                entries.append((point.key(), workload))
            if base.kernel.sources:
                for key, workload in entries:
                    if workload.qualifies(min_speedup) or base.kernel.same_kernel(schema.kernel):
                        served.setdefault(key, []).append((base, workload, path))
        incumbents = {}
        for workload in schema.workloads:
            key = schema.point_with_contract_id(workload.axes, contract.contract_id).key()
            if key in served:
                base, served_workload, path = max(
                    served[key],
                    key=lambda entry: (
                        entry[0].kernel.same_kernel(schema.kernel),
                        entry[0].rank_key(entry[1]),
                    ),
                )
                incumbents[axes_text(workload.axes)] = (
                    base,
                    path,
                    served_workload.qualifies(min_speedup),
                )
        return incumbents

    def _run_batch(
        self,
        scope: ValidationScope,
        executor: Executor,
        batch: tuple[TrialRequest, ...],
        done: Iterator[int],
        total: int,
    ) -> tuple[list[Finding], list[WorkloadRecord]]:
        failures: list[Finding] = []
        candidate = reference = None
        candidate_timeout = BUILD_TIMEOUT_SECONDS + WORKLOAD_TIMEOUT_SECONDS * len(batch)
        try:
            future = executor.submit(
                run_trial,
                # the candidate twice (a tracer fault reruns it), each in a child it starts
                timeout_seconds=2 * (DEFAULT_STARTUP_TIMEOUT_SECONDS + candidate_timeout)
                + _evaluation_timeout_seconds(batch),
                requests=batch,
                candidate_timeout_seconds=candidate_timeout,
                n_cpu_shares=len(self._devices),
            )
            trial = TrialResult.from_task(future.result())
            candidate, reference = trial.candidate, trial.reference
        except (ExecutionError, ValueError) as e:
            # the candidate's failures are in the result: the trial's own are the platform's
            detail = str(e).strip().split("\n")[-1]
            failures.append(Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"trial: {detail}"))
        findings, records = self._judged_findings(scope, batch, failures, candidate, reference)
        try:
            line = BenchReport.from_findings(findings, workload_records=tuple(records)).batch_line()
        except Exception:  # a progress line never costs the batch's records
            logging.getLogger(__name__).exception("bench progress line failed")
            line = f"{records[0].schema_path}: progress text unavailable"
        self.progress.write(f"{next(done)}/{total} {line}")
        return findings, records

    def _judged_findings(
        self,
        scope: ValidationScope,
        batch: tuple[TrialRequest, ...],
        failures: Sequence[Finding],
        candidate: CandidateResult | None,
        reference: Sequence[ReferenceWorkloadResult] | None,
    ) -> tuple[list[Finding], list[WorkloadRecord]]:
        batch_findings = [*failures, *(candidate.findings if candidate else ())]
        if not batch_findings:
            candidate_results = {r.axes_text: r for r in candidate.workload_results}
            reference_results = {r.axes_text: r for r in reference}
        schema_path = scope.relative_path(Path(batch[0].schema_path))
        findings: list[Finding] = []
        records: list[WorkloadRecord] = []
        for request in batch:
            found, timings, worst_violation_fraction = list(batch_findings), (), None
            if not batch_findings:
                candidate_result = candidate_results[request.axes_text]
                # absent: declined by the baseline, the candidate's finding alone
                reference_result = reference_results.get(request.axes_text)
                found = list(candidate_result.findings)
                if reference_result is not None:
                    worst_violation_fraction = reference_result.worst_violation_fraction
                    found[:0] = reference_result.findings
                    if not candidate_result.findings:
                        if not reference_result.is_complete:
                            detail = "outputs of steps are missing, and no finding says why"
                            found.append(Finding.unlocated(FindingCode.CANDIDATE_CRASH, detail))
                        timings, gated = _measure.compare_performance(
                            candidate_result.rounds, request.thresholds
                        )
                        if gated is not None:
                            found.append(gated)
            located = {"schema_path": schema_path, "axes": request.workload.axes}
            # a defined baseline is named once it timed; untimed: nothing to export
            baseline = request.baseline
            if baseline is not None and not baseline.endswith(".json"):
                baseline = None if batch_findings or not candidate_result.rounds else baseline
            findings += [f.model_copy(update=located) for f in found]
            records.append(
                WorkloadRecord(
                    **located,
                    batch_stopped=bool(batch_findings),
                    kernel_content_hash=request.schema.kernel.content_hash,
                    baseline=baseline,
                    is_timed="baseline" in request.thresholds,
                    worst_violation_fraction=worst_violation_fraction,
                    timings=timings,
                )
            )
        return findings, records
