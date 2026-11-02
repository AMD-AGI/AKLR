###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import itertools
import json
import logging
import shutil
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, TypeVar

from pydantic import TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.schema import Schema, load_json_file
from aklr.core.schema._workload import Workload
from aklr.core.validation.stages import _measure
from aklr.core.validation.stages._bench._evaluation import (
    CandidateResult,
    ReferenceResult,
    TrialRequest,
    run_candidate,
    run_reference,
)
from aklr.core.validation.stages._bench._execution import (
    ExecutionError,
    Executor,
    TaskExecutionTimeout,
    caused_by_the_task,
)
from aklr.core.validation.stages._bench._report import BenchReport, WorkloadRecord
from aklr.core.validation.stages._measure import Threshold, Versus
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages._report._markdown import axes_text
from aklr.core.validation.stages.stage import PipelineStage
from aklr.core.validation.stages.validation_scope import ValidationScope

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport

__all__ = [
    "BUILD_TIMEOUT_SECONDS", "MAX_BATCH_OUTPUT_BYTES", "MAX_BATCH_WORKLOADS",
    "REFERENCE_SECONDS_PER_OUTPUT_GIB", "WORKLOAD_TIMEOUT_SECONDS",
    "BenchStage", "ReferenceNotProduced",
]  # fmt: skip

MAX_BATCH_OUTPUT_BYTES: Final = 16 << 30
# one batch runs on one device: a long schema split up spreads over all of them
MAX_BATCH_WORKLOADS: Final = 16
BUILD_TIMEOUT_SECONDS: Final = 900.0
WORKLOAD_TIMEOUT_SECONDS: Final = 300.0
# the float64 CPU reference scales with outputs: gemm_n3072_k512 at m=342073, 23 GiB, ~420 s on a busy MI355X
REFERENCE_SECONDS_PER_OUTPUT_GIB: Final = 60.0
ResultT = TypeVar("ResultT", CandidateResult, ReferenceResult)


class ReferenceNotProduced(RuntimeError):
    """The reference task failed or gave no result: the platform's fault, never the candidate's."""


class UnreadableIncumbent(RuntimeError):
    """A same-name base schema that is not even JSON: its replacement is never timed without it."""


class BenchStage(PipelineStage):
    """Each schema's workloads in batches: a candidate task, then a reference task, per device."""

    name = "bench"
    report_type = BenchReport

    def __init__(self, devices: Sequence[str]) -> None:
        if not devices:
            raise ValueError("no device to run on")
        self._devices = tuple(devices)
        # by (base sha, path): every benched schema of an op reads all of its op's base files
        self._base_schemas: dict[tuple[str, str], Schema | ValueError | None] = {}

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> BenchReport:
        if not scope.bench_paths:
            return BenchReport.skipped(self.name, "no kernel to measure")
        self.progress.write(f"reading {len(scope.bench_paths)} schemas")
        unreadable: list[Finding] = []
        batches: list[tuple[TrialRequest, ...]] = []
        target = DeviceTarget.from_names(earlier.platform.arch, earlier.platform.hardware)
        for path in scope.bench_paths:
            try:
                batches += self._workloads_to_run(scope, path, target)
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
        # costliest first, so none is the tail: ~1 unit per workload, +1 per GiB of inputs redrawn each step
        order = sorted(
            range(len(batches)),
            key=lambda i: -sum(1 + r.input_bytes / 2**30 for r in batches[i]),
        )
        with (
            Executor.for_devices(self._devices) as executor,
            ThreadPoolExecutor(len(self._devices)) as pool,
        ):
            started = list(
                pool.map(
                    lambda i: self._run_batch(scope, executor, batches[i], done, len(batches)),
                    order,
                )
            )
        outcomes = [started[order.index(i)] for i in range(len(batches))]
        findings = [*unreadable, *(f for found, _ in outcomes for f in found)]
        records = [r for _, batch_records in outcomes for r in batch_records]
        return BenchReport.from_findings(self.name, findings, workload_records=tuple(records))

    def _workloads_to_run(
        self, scope: ValidationScope, path: Path, target: DeviceTarget
    ) -> list[tuple[TrialRequest, ...]]:
        base_sha = scope.base_sha
        path = path.resolve()
        schema_path = path.relative_to(scope.root.resolve()).as_posix()
        schema = load_json_file(Schema, path)
        if schema.kernel is None:
            return []
        import aklr.seam  # noqa: F401 -- declares every adapter
        from aklr.core.runtime.seam import SeamRegistry

        adapters = SeamRegistry.default().adapters(schema.op_type)
        # speedups are keyed by the backend: the first adapter's top-level package
        backend = adapters[0].backend if adapters else ""
        # dispatch's rule: a kernel must beat the backend, if there is one
        min_speedup = RuntimeEnv().min_speedup if adapters else 0
        incumbents = (
            self._incumbents(scope, schema_path, schema, target, min_speedup, backend) if base_sha else {}
        )
        # A first release has no incumbent; per-kernel admission is in BenchReport.
        first_release = scope.is_first_release
        content_hash = schema.kernel.hash()
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
            incumbent, incumbent_path, is_served = incumbents.get(workload_axes_text, (None, None, False))
            is_kept = (
                not base_sha or first_release or (incumbent is not None and incumbent.kernel == schema.kernel)
            )
            thresholds: dict[Versus, Threshold] = {"baseline": "retain" if is_kept else "entry"}
            if incumbent is not None and incumbent.kernel.hash() != content_hash:
                thresholds["incumbent"] = "retain"
            else:
                incumbent = None  # this very kernel: never timed against itself
            if not adapters:
                # the incumbent times as the baseline; with none, nothing is timed
                thresholds = {"baseline": "retain"} if incumbent is not None else {}
            request = TrialRequest.of_workload(
                str(path),
                schema,
                workload,
                thresholds=thresholds,
                incumbent=incumbent,
                baseline=incumbent_path if is_served else None,
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
        schema: Schema,
        target: DeviceTarget,
        min_speedup: float,
        backend: str,
    ) -> dict[str, tuple[Schema, str, bool]]:
        impls_dir = (Path(schema_path).parents[3] / "impls").as_posix()
        paths = [p for p in scope.paths_at_base(impls_dir) if ValidationScope.path_kind(p) == "schema_data"]
        served: dict[Any, list[tuple[Schema, Workload, str]]] = {}
        for path in paths:
            key = (scope.base_sha, path)
            if key not in self._base_schemas:
                text = scope.text_at_base(path)
                try:
                    base = TypeAdapter(Schema).validate_json(text)
                except ValueError as e:
                    base = e
                    try:  # JSON, so base CI read it: a schema of another schema format
                        json.loads(text)
                        base = None
                    except ValueError:
                        pass
                self._base_schemas[key] = base
            base = self._base_schemas[key]
            if base is None:
                continue
            if isinstance(base, ValueError):
                if PurePosixPath(path).parent.name != schema.name:
                    continue
                raise UnreadableIncumbent(f"cannot read the incumbent {path}: {base}") from base
            if base.kernel is None:
                continue
            for workload in base.workloads:
                if workload.qualifies(target, min_speedup, backend) or base.kernel == schema.kernel:
                    workload_key = (base.name, tuple(sorted(workload.axes.items())))
                    served.setdefault(workload_key, []).append((base, workload, path))
        incumbents = {}
        for workload in schema.workloads:
            workload_key = (schema.name, tuple(sorted(workload.axes.items())))
            if workload_key in served:
                base, served_workload, path = max(
                    served[workload_key],
                    key=lambda entry: (
                        entry[0].kernel == schema.kernel,
                        (e := entry[1].evaluation(target)) is None or e.accuracy != "failed",
                        (
                            e.speedup[backend].value
                            if e is not None and backend in e.speedup
                            else float("-inf")
                        ),
                        entry[0].kernel.hash(),
                    ),
                )
                incumbents[axes_text(workload.axes)] = (
                    base,
                    path,
                    served_workload.qualifies(target, min_speedup, backend),
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
        directories: list[Path] = []
        candidate = reference = None
        # a succeeded task's directory is the caller's to delete
        is_succeeded = lambda f: f.done() and not f.cancelled() and f.exception() is None
        try:
            future = None
            try:
                future = executor.submit(
                    run_candidate,
                    timeout_seconds=BUILD_TIMEOUT_SECONDS + WORKLOAD_TIMEOUT_SECONDS * len(batch),
                    requests=batch,
                )
                candidate = self._task_result(CandidateResult, future.result(), batch)
            except (ExecutionError, ValueError) as e:
                failures.append(self._build_failed_finding(e))
            finally:
                if future is not None and is_succeeded(future) and future.directory:
                    directories.append(future.directory)
            if candidate is not None and not candidate.findings:
                declined = {
                    r.axes_text
                    for r in candidate.workload_results
                    if any(f.code == FindingCode.BASELINE_FAILED for f in r.findings)
                }
                requests = tuple(r for r in batch if r.axes_text not in declined)
                reference = ReferenceResult(workload_results=())
                if requests:
                    reference_future = None
                    try:
                        reference_future = executor.submit(
                            run_reference,
                            timeout_seconds=WORKLOAD_TIMEOUT_SECONDS * len(requests)
                            + REFERENCE_SECONDS_PER_OUTPUT_GIB
                            * sum(r.output_bytes for r in requests)
                            / (1 << 30),
                            requests=requests,
                            candidate_dir=future.directory,
                        )
                        try:
                            reference = self._task_result(
                                ReferenceResult, reference_future.result(), requests
                            )
                        except ExecutionError as e:
                            raise ReferenceNotProduced(f"reference: {self._last_line(e)}") from e
                        except ValueError as e:
                            raise ReferenceNotProduced(f"internal: {e}") from e
                    except (ExecutionError, ReferenceNotProduced) as e:
                        reference = None
                        failures.append(self._build_failed_finding(e))
                    finally:
                        if (
                            reference_future is not None
                            and is_succeeded(reference_future)
                            and reference_future.directory
                        ):
                            directories.append(reference_future.directory)
        finally:
            for directory in directories:
                shutil.rmtree(directory, ignore_errors=True)
        findings, records = self._judged_findings(scope, batch, failures, candidate, reference)
        try:
            line = BenchReport.from_findings(
                self.name, findings, workload_records=tuple(records)
            ).batch_line()
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
        reference: ReferenceResult | None,
    ) -> tuple[list[Finding], list[WorkloadRecord]]:
        batch_findings = [*failures, *(candidate.findings if candidate else ())]
        if not batch_findings:
            candidate_results = {r.axes_text: r for r in candidate.workload_results}
            reference_results = {r.axes_text: r for r in reference.workload_results}
        schema_path = scope.relative_path(Path(batch[0].schema_path))
        findings: list[Finding] = []
        records: list[WorkloadRecord] = []
        for request in batch:
            found, timings, is_compared, backend = list(batch_findings), (), False, None
            if not batch_findings:
                candidate_result = candidate_results[request.axes_text]
                # absent: declined by the baseline (BASELINE_FAILED); the candidate's finding stands alone
                reference_result = reference_results.get(request.axes_text)
                found = list(candidate_result.findings)
                backend = candidate_result.backend
                if reference_result is not None:
                    is_compared = reference_result.is_compared
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
            baseline = request.baseline or (None if batch_findings else candidate_result.baseline)
            findings += [replace(f, **located) for f in found]
            records.append(
                WorkloadRecord(
                    **located,
                    batch_stopped=bool(batch_findings),
                    kernel_content_hash=request.schema.kernel.hash(),
                    baseline=baseline,
                    backend=backend,
                    is_timed="baseline" in request.thresholds,
                    is_compared=is_compared,
                    timings=timings,
                )
            )
        return findings, records

    def _build_failed_finding(self, error: BaseException) -> Finding:
        detail = self._last_line(error)
        if isinstance(error, TaskExecutionTimeout):
            code = FindingCode.CANDIDATE_TIMEOUT
        elif caused_by_the_task(error) or isinstance(error, ValueError):
            code = FindingCode.CANDIDATE_CRASH
        else:
            code = FindingCode.PLATFORM_FAILURE
        return Finding.unlocated(code, detail)

    @staticmethod
    def _last_line(error: BaseException) -> str:
        return str(error).strip().split("\n")[-1]

    @staticmethod
    def _task_result(result_type: type[ResultT], value: Any, requests: Sequence[TrialRequest]) -> ResultT:
        detail = f"the task's result is not a {result_type.__name__} of the batch"
        try:
            result = TypeAdapter(result_type).validate_python(value)
        except ValueError as e:
            raise ValueError(detail) from e
        axes_texts = sorted(r.axes_text for r in result.workload_results)
        if axes_texts != sorted(r.axes_text for r in requests):
            raise ValueError(detail)
        return result
