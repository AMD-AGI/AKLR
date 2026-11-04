# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The trial task of a batch: its process lends tensors for the outputs to a child task
that runs the candidate, then compares what the child left in them with the reference,
all on one device, so that no output goes through the host and no candidate code runs
where the reference does. An op that ranks call together takes a child task a rank, each
on its device of the group the trial task leased."""

from __future__ import annotations

import concurrent.futures
import contextlib
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages import _measure
from aklr.core.validation.stages._bench._evaluation._candidate import (
    CandidateResult,
    run_candidate,
)
from aklr.core.validation.stages._bench._evaluation._reference import (
    ReferenceWorkloadResult,
    WorkloadEvaluation,
)
from aklr.core.validation.stages._bench._evaluation._request import TrialRequest
from aklr.core.validation.stages._bench._execution import (
    ExecutionContext,
    ExecutionError,
    Executor,
    ProcessGroup,
)
from aklr.core.validation.stages._report._finding import FindingCode

__all__ = ["TrialResult", "run_trial"]


class TrialResult(FrozenModel):
    candidate: CandidateResult
    # one per workload its baseline did not decline; None: the candidate stopped the batch
    reference: tuple[ReferenceWorkloadResult, ...] | None

    @classmethod
    def from_task(cls, value: Any) -> TrialResult:
        detail = "internal: the trial task's result is not one for the batch"
        try:
            result = cls.model_validate(value)
        except ValueError as e:
            raise ValueError(detail) from e
        candidate, reference = result.candidate, result.reference
        # the candidate's workloads the trial task checked (CandidateResult.from_task)
        evaluated = [r.axes_text for r in candidate.workload_results if not r.is_declined]
        if candidate.findings:
            is_whole = reference is None
        else:
            is_whole = reference is not None and sorted(r.axes_text for r in reference) == sorted(evaluated)
        if not is_whole:
            raise ValueError(detail)
        return result


def run_trial(
    context: ExecutionContext,
    *,
    requests: Sequence[TrialRequest],
    candidate_timeout_seconds: float,
    n_cpu_shares: int,
) -> dict:
    """Run the batch's candidate in a child task on ``context.device``, or one a rank on
    each of the group's devices (a tracer fault reruns it once), lent
    ``TrialRequest.empty_outputs``, then evaluate each workload its baseline did not
    decline; returns ``TrialResult`` json."""
    import torch

    from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX

    axes_texts = [r.axes_text for r in requests]
    is_group = requests[0].contract.ranks is not None
    if is_group and context.n_devices != requests[0].n_ranks:
        raise RuntimeError(f"{requests[0].n_ranks} ranks call the op together, {context.n_devices} devices")
    devices = [context.device]
    if is_group:
        devices = (
            [context.device] * context.n_devices
            if context.device == "cpu"
            else [f"cuda:{r}" for r in range(context.n_devices)]
        )
    outputs = []
    for device in devices:
        # in a pool of their own: a lent tensor's IPC handle maps its whole allocator segment, which
        # no tensor of the reference's may share
        pool = None if device == "cpu" else torch.cuda.MemPool()
        with contextlib.nullcontext() if pool is None else torch.cuda.use_mem_pool(pool, device=device):
            outputs.append({r.axes_text: r.empty_outputs(device) for r in requests})
    try:
        for attempt in range(2):
            # a rank's finding or failure ends the group: its peers would wait on it forever
            # a group's every task leases all its devices, in the trial task's order (devices
            # repeated, a group a rank): it receives tensors lent on any of them
            with Executor.for_devices(
                devices * len(devices) if is_group else devices,
                devices_per_task=len(devices) if is_group else 1,
                n_cpu_shares=n_cpu_shares,
            ) as executor:
                # a FileStore's file meets one group once: a fresh one an attempt
                group = (
                    {"rendezvous": str(Path(tempfile.mkdtemp(prefix="group-")) / "store")} if is_group else {}
                )
                futures = {
                    executor.submit(
                        run_candidate,
                        timeout_seconds=candidate_timeout_seconds,
                        requests=requests,
                        outputs=outputs[rank],
                        **({"rank": rank} | group if is_group else {}),
                    ): rank
                    for rank in range(len(devices))
                }
                by_rank: dict[int, CandidateResult] = {}
                for future in concurrent.futures.as_completed(futures):
                    rank = futures[future]
                    try:
                        result = CandidateResult.from_task(future.result(), axes_texts)
                    except (ExecutionError, ValueError) as e:
                        result = CandidateResult.of_failure(e)
                    if is_group and result.findings:
                        findings = tuple(
                            f.model_copy(update={"detail": f"rank {rank}: {f.detail}"})
                            for f in result.findings
                        )
                        result = result.model_copy(update={"findings": findings})
                    by_rank[rank] = result
                    if result.findings:
                        executor.shutdown(wait=False)
                        break
            per_rank = [by_rank[r] for r in range(len(devices))] if len(by_rank) == len(devices) else []
            failed = [r for r in by_rank.values() if r.findings]
            candidate = failed[0] if failed else CandidateResult.of_ranks(per_rank)
            # a tracer fault is the platform's: fresh processes measure again
            if attempt == 0 and failed and candidate.findings[0].code == FindingCode.PLATFORM_FAILURE:
                if _measure.is_tracer_fault(candidate.findings[0].detail):
                    continue
            break
    finally:
        # a candidate's process that left its session outlives the executor's kills
        if not ProcessGroup.kill_adopted():
            raise RuntimeError("the candidate's processes outlived their kill")
    reference = None
    if not candidate.findings:
        by_axes = [{w.axes_text: w for w in r.workload_results} for r in per_rank]
        reference = tuple(
            WorkloadEvaluation(
                request,
                [o[request.axes_text] for o in outputs],
                [w[request.axes_text].output_keys for w in by_axes],
                devices,
            ).evaluate()
            for request in requests
            if not by_axes[0][request.axes_text].is_declined
        )
    if kernels := [m for m in sys.modules if m.startswith(KERNEL_PACKAGE_PREFIX)]:
        raise RuntimeError(f"the trial task imported kernels: {sorted(kernels)}")
    return TrialResult(candidate=candidate, reference=reference).model_dump(mode="json")
