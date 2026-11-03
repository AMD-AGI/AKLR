# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aklr.core.contract import semantics
from aklr.core.contract.locate import op_module
from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._bench._evaluation._comparison import (
    find_nondeterminism,
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


class ReferenceWorkloadResult(FrozenModel):
    axes_text: str
    findings: tuple[Finding, ...]
    worst_violation_fraction: float | None
    is_complete: bool


class ReferenceResult(FrozenModel):
    workload_results: tuple[ReferenceWorkloadResult, ...]

    @classmethod
    def from_task(cls, value: Any, axes_texts: Sequence[str]) -> ReferenceResult:
        detail = "internal: the reference task's result is not a ReferenceResult"
        try:
            result = cls.model_validate(value)
        except ValueError as e:
            raise ValueError(detail) from e
        if sorted(r.axes_text for r in result.workload_results) != sorted(axes_texts):
            raise ValueError(detail)
        return result


def run_reference(
    context: ExecutionContext,
    *,
    requests: Sequence[TrialRequest],
    candidate_dir: Path,
) -> dict:
    """Compare the batch's results with the float64 reference; returns ``ReferenceResult`` json."""
    from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(candidate_dir, flags)
    try:
        workload_results = tuple(
            WorkloadEvaluation(request, directory_descriptor, context.device).evaluate()
            for request in requests
        )
    finally:
        os.close(directory_descriptor)
    if kernels := [m for m in sys.modules if m.startswith(KERNEL_PACKAGE_PREFIX)]:
        raise RuntimeError(f"the reference task imported kernels: {sorted(kernels)}")
    return ReferenceResult(workload_results=workload_results).model_dump(mode="json")


class WorkloadEvaluation:
    """One workload's candidate results against the float64 reference."""

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
        contract, point = request.contract, request.point
        steps = list(request.steps)
        if step := next((s for s in steps if s.input_set == "workload"), None):
            try:
                request.compose_inputs(step)
            except ValueError as e:
                self._fail(step, FindingCode.SCHEMA_INVALID, str(e))
                steps.remove(step)
        file_name = outputs_file_name(request.axes_text)
        try:
            candidate_file = open_candidate_outputs(self.directory_descriptor, file_name)
            names = set(candidate_file.keys())
        except (OSError, ValueError, SafetensorError) as e:
            self._fail(None, FindingCode.CANDIDATE_CRASH, f"{file_name}: {e}")
            return self._result(None, is_complete=False)

        n_present = 0
        while n_present < len(steps) and names.issuperset(
            steps[n_present].output_key(o) for o in contract.mutable_tensors
        ):
            n_present += 1
        present_keys = {s.output_key(o) for s in steps[:n_present] for o in contract.mutable_tensors}
        if extra := names - present_keys:
            detail = f"{file_name}: {sorted(extra)} are not outputs before a missing one"
            self._fail(None, FindingCode.CANDIDATE_CRASH, detail)
            return self._result(None, is_complete=False)

        mutable_json = semantics.mutable_tensor_declarations(request.schema_json)
        mutable_shapes = {r: contract.storage_shape_of(r, point) for r in contract.mutable_tensors}
        mutable_dtypes = {r: point.dtypes[r] for r in contract.mutable_tensors}
        reference_module = op_module(contract.op_package, "reference")
        reference_fn = getattr(reference_module, contract.reference_name)
        worst = None
        passed_mutable: dict[str, dict[str, torch.Tensor]] = {}
        steps_by_inputs: dict[str, list[Step]] = {}
        for step in steps[:n_present]:
            steps_by_inputs.setdefault(step.input_set, []).append(step)
        for (
            first,
            *rest,
        ) in steps_by_inputs.values():
            row_major_args = request.compose_inputs(first)
            row_major_args |= {n: v.to(device) for n, v in row_major_args.items() if torch.is_tensor(v)}
            compare_kwargs = {
                "reference_outputs": contract.mutable_tensors_by_name(
                    reference_fn(point, row_major_args, semantics.to_float64)
                )
            }
            if request.has_error_bound("dot_product"):
                magnitudes = {
                    n: semantics.absolute_value(v, point.dtypes[n])
                    for n, v in row_major_args.items()
                    if torch.is_tensor(v)
                }
                compare_kwargs["magnitude_outputs"] = contract.mutable_tensors_by_name(
                    reference_fn(point, row_major_args | magnitudes, semantics.to_float64)
                )
            if request.has_error_bound("baseline_relative"):
                baseline = contract.baseline()
                if baseline is None:
                    raise ValueError(f"{contract.contract_id}: a baseline_relative bound needs a baseline")
                call = request.assemble_call(row_major_args, device)
                compare_kwargs["baseline_outputs"] = contract.call_mutable_tensors(baseline(**call), call)
            compare_kwargs = {
                kind: {r: t.cpu() for r, t in tensors.items()} for kind, tensors in compare_kwargs.items()
            }
            compared = functools.partial(semantics.compare_outputs, request.schema_json, **compare_kwargs)
            if not is_reference_comparable(mutable_dtypes, compare_kwargs["reference_outputs"], compared):
                detail = "the reference rounded to the output dtypes is not finite or fails compare"
                if first.input_set == "workload":
                    self._fail(first, FindingCode.SCHEMA_INVALID, detail)
                else:
                    self._fail(first, FindingCode.PLATFORM_FAILURE, f"internal: {detail}")
                continue
            for step in (first, *rest):
                try:
                    actual_mutable = read_step_outputs(candidate_file, step, mutable_shapes, mutable_json)
                except ValueError as e:
                    self._fail(step, FindingCode.CANDIDATE_CRASH, str(e))
                    continue
                compare_results = compared(actual_mutable)
                worst = max(
                    worst or 0.0,
                    *(r.violation_fraction for r in compare_results.values()),
                )
                if failed := {o: r for o, r in compare_results.items() if not r.is_passed}:
                    details = (f"{o}: {r.detail}" for o, r in failed.items())
                    self._fail(step, step.mismatch_code, "; ".join(details))
                elif step.name in ("numeric/0", "determinism/0"):
                    passed_mutable[step.name] = actual_mutable
        if len(passed_mutable) == 2:
            detail = find_nondeterminism(
                request.schema_json,
                passed_mutable["numeric/0"],
                passed_mutable["determinism/0"],
            )
            if detail is not None:
                step = next(s for s in steps if s.check == "determinism")
                self._fail(step, FindingCode.NONDETERMINISTIC, detail)
        return self._result(worst, is_complete=n_present == len(steps))

    def _fail(self, step: Step | None, code: FindingCode, detail: str) -> None:
        if step is None:
            self._failures.append((-1, -1, Finding.unlocated(code, detail)))
            return
        position = self.request.steps.index(step)
        key = position if code == FindingCode.SCHEMA_INVALID else step.check
        finding = Finding.unlocated(code, f"{step.label}: {detail}")
        self._failures.append((position, key, finding))

    def _result(self, worst_violation_fraction: float | None, is_complete: bool) -> ReferenceWorkloadResult:
        firsts: dict[str | int, Finding] = {}
        for _, key, finding in sorted(self._failures, key=lambda f: f[0]):
            firsts.setdefault(key, finding)
        return ReferenceWorkloadResult(
            axes_text=self.request.axes_text,
            findings=tuple(firsts.values()),
            worst_violation_fraction=worst_violation_fraction,
            is_complete=is_complete,
        )
