# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The trial task of a batch: its process lends tensors for the outputs to a child task
that runs the candidate, then compares what the child left in them with the reference,
all on one device, so that no output goes through the host and no candidate code runs
where the reference does."""

from __future__ import annotations

import sys
from collections.abc import Sequence
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
    """Run the batch's candidate in a child task on ``context.device`` (a tracer fault
    reruns it once), lent ``TrialRequest.empty_outputs``, then evaluate each workload its
    baseline did not decline; returns ``TrialResult`` json."""
    from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX

    axes_texts = [r.axes_text for r in requests]
    outputs = {r.axes_text: r.empty_outputs(context.device) for r in requests}
    with Executor.for_devices([context.device], n_cpu_shares=n_cpu_shares) as executor:
        for attempt in range(2):
            try:
                future = executor.submit(
                    run_candidate,
                    timeout_seconds=candidate_timeout_seconds,
                    requests=requests,
                    outputs=outputs,
                )
                candidate = CandidateResult.from_task(future.result(), axes_texts)
            except (ExecutionError, ValueError) as e:
                candidate = CandidateResult.of_failure(e)
                (failure,) = candidate.findings
                # a tracer fault is the platform's: a fresh process measures again
                if (
                    attempt == 0
                    and failure.code == FindingCode.PLATFORM_FAILURE
                    and _measure.is_tracer_fault(failure.detail)
                ):
                    continue
            break
    reference = None
    if not candidate.findings:
        workload_results = {r.axes_text: r for r in candidate.workload_results}
        reference = tuple(
            WorkloadEvaluation(
                r,
                outputs[r.axes_text],
                workload_results[r.axes_text].output_keys,
                context.device,
            ).evaluate()
            for r in requests
            if not workload_results[r.axes_text].is_declined
        )
    if kernels := [m for m in sys.modules if m.startswith(KERNEL_PACKAGE_PREFIX)]:
        raise RuntimeError(f"the trial task imported kernels: {sorted(kernels)}")
    return TrialResult(candidate=candidate, reference=reference).model_dump(mode="json")
