###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.validation.stages._bench._evaluation._comparison import (
    is_reference_comparable,
)
from aklr.core.validation.stages._bench._evaluation._safetensors import (
    open_candidate_outputs,
    outputs_file_name,
    read_step_outputs,
)
from aklr.core.validation.stages._bench._evaluation._trials import Step, TrialRequest
from aklr.core.validation.stages._bench._execution import ExecutionContext
from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    import torch

__all__ = [
    "ReferenceResult", "ReferenceWorkloadResult", "WorkloadEvaluation",
    "run_reference",
]  # fmt: skip


@frozen_dataclass
class ReferenceWorkloadResult:
    axes_text: str
    findings: tuple[Finding, ...]
    is_compared: bool
    is_complete: bool


@frozen_dataclass
class ReferenceResult:
    workload_results: tuple[ReferenceWorkloadResult, ...]


def run_reference(
    context: ExecutionContext,
    *,
    requests: Sequence[TrialRequest],
    candidate_dir: Path,
) -> dict:
    """Compare the batch's outputs with the float64 reference; returns ``ReferenceResult`` json."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(candidate_dir, flags)
    try:
        workload_results = tuple(
            WorkloadEvaluation(request, directory_descriptor, context.device).evaluate()
            for request in requests
        )
    finally:
        os.close(directory_descriptor)
    return TypeAdapter(ReferenceResult).dump_python(
        ReferenceResult(workload_results=workload_results), mode="json"
    )


class WorkloadEvaluation:
    request: TrialRequest
    directory_descriptor: int
    device: str | torch.device
    _failures: list[tuple[int, str | int, Finding]]

    def __init__(
        self,
        request: TrialRequest,
        directory_descriptor: int,
        device: str | torch.device,
    ) -> None:
        self.request = request
        self.directory_descriptor = directory_descriptor
        self.device = device
        self._failures = []

    def evaluate(self) -> ReferenceWorkloadResult:
        import torch
        from safetensors import SafetensorError

        request, device = self.request, self.device
        schema = request.schema
        steps = list(request.steps)
        file_name = outputs_file_name(request.axes_text)
        try:
            candidate_file = open_candidate_outputs(self.directory_descriptor, file_name)
            names = set(candidate_file.keys())
        except (OSError, ValueError, SafetensorError) as e:
            self._fail(None, FindingCode.CANDIDATE_CRASH, f"{file_name}: {e}")
            return self._result(False, is_complete=False)

        n_present = 0
        while n_present < len(steps) and names.issuperset(
            steps[n_present].output_key(o) for o in schema.outputs
        ):
            n_present += 1
        present_keys = {s.output_key(o) for s in steps[:n_present] for o in schema.outputs}
        if extra := names - present_keys:
            detail = f"{file_name}: {sorted(extra)} are not outputs before a missing one"
            self._fail(None, FindingCode.CANDIDATE_CRASH, detail)
            return self._result(False, is_complete=False)

        output_shapes = request.output_shapes
        output_dtypes = {o: spec.dtype for o, spec in schema.outputs.items()}
        reference_fn = schema.source_to_function("reference")
        check = schema.source_to_function("check")
        reference_params = inspect.signature(reference_fn).parameters
        is_compared = n_present > 0
        steps_by_inputs: dict[str, list[Step]] = {}
        for step in steps[:n_present]:
            steps_by_inputs.setdefault(step.input_set, []).append(step)
        for first, *rest in steps_by_inputs.values():
            call = request.compose_call(first, device)
            with torch.no_grad():
                result = reference_fn(**{n: v for n, v in call.items() if n in reference_params})
            results = result if isinstance(result, (tuple, list)) else (result,)
            reference_outputs = {o: t.cpu() for o, t in zip(schema.outputs, results, strict=True)}
            del call
            if not is_reference_comparable(output_dtypes, reference_outputs, check):
                detail = "the reference rounded to the output dtypes is not finite or fails check"
                self._fail(first, FindingCode.PLATFORM_FAILURE, f"internal: {detail}")
                is_compared = False
                continue
            for step in (first, *rest):
                try:
                    actual_outputs = read_step_outputs(candidate_file, step, output_shapes, output_dtypes)
                except ValueError as e:
                    self._fail(step, FindingCode.CANDIDATE_CRASH, str(e))
                    continue
                if not check(list(reference_outputs.values()), list(actual_outputs.values())):
                    self._fail(step, step.mismatch_code, "the outputs fail the check")
        return self._result(is_compared, is_complete=n_present == len(steps))

    def _fail(self, step: Step | None, code: FindingCode, detail: str) -> None:
        if step is None:
            self._failures.append((-1, -1, Finding.unlocated(code, detail)))
            return
        position = self.request.steps.index(step)
        key = position if code == FindingCode.SCHEMA_INVALID else step.check
        finding = Finding.unlocated(code, f"{step.label}: {detail}")
        self._failures.append((position, key, finding))

    def _result(self, is_compared: bool, is_complete: bool) -> ReferenceWorkloadResult:
        firsts: dict[str | int, Finding] = {}
        for _, key, finding in sorted(self._failures, key=lambda f: f[0]):
            firsts.setdefault(key, finding)
        return ReferenceWorkloadResult(
            axes_text=self.request.axes_text,
            findings=tuple(firsts.values()),
            is_compared=is_compared,
            is_complete=is_complete,
        )
