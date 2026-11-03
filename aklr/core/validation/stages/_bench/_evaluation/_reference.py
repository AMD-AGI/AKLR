# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from aklr.core.contract import semantics
from aklr.core.contract.locate import op_module
from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._bench._evaluation._comparison import (
    find_nondeterminism,
    is_reference_comparable,
)
from aklr.core.validation.stages._bench._evaluation._trials import Step, TrialRequest
from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    import torch

__all__ = ["ReferenceWorkloadResult", "WorkloadEvaluation"]


class ReferenceWorkloadResult(FrozenModel):
    axes_text: str
    findings: tuple[Finding, ...]
    worst_violation_fraction: float | None
    is_complete: bool


class WorkloadEvaluation:
    """One workload's outputs, as the candidate left them in the lent tensors, against the
    float32 reference, on their device."""

    request: TrialRequest
    outputs: Mapping[str, torch.Tensor]
    output_keys: Sequence[str]
    device: str | torch.device
    _failures: list[tuple[int, str | int, Finding]]

    def __init__(
        self,
        request: TrialRequest,
        outputs: Mapping[str, torch.Tensor],
        output_keys: Sequence[str],
        device: str | torch.device,
    ) -> None:
        """``outputs``: ``request.empty_outputs``, of which the candidate filled
        ``output_keys``."""
        self.request = request
        self.outputs = outputs
        self.output_keys = output_keys
        self.device = device
        self._failures = []

    def evaluate(self) -> ReferenceWorkloadResult:
        import torch

        request, device = self.request, self.device
        contract, point = request.contract, request.point
        steps = list(request.steps)
        if step := next((s for s in steps if s.input_set == "workload"), None):
            try:
                request.compose_inputs(step, device)
            except ValueError as e:
                self._fail(step, FindingCode.SCHEMA_INVALID, str(e))
                steps.remove(step)
        names = set(self.output_keys)
        n_present = 0
        while n_present < len(steps) and names.issuperset(
            steps[n_present].output_key(o) for o in contract.mutable_tensors
        ):
            n_present += 1
        present_keys = {s.output_key(o) for s in steps[:n_present] for o in contract.mutable_tensors}
        if extra := names - present_keys:
            detail = f"{sorted(extra)} are not outputs before a missing one"
            self._fail(None, FindingCode.CANDIDATE_CRASH, detail)
            return self._result(None, is_complete=False)

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
            row_major_args = request.compose_inputs(first, device)
            compare_kwargs = {
                "reference_outputs": contract.mutable_tensors_by_name(
                    reference_fn(point, row_major_args, semantics.to_float32)
                )
            }
            if request.has_error_bound("dot_product"):
                magnitudes = {
                    n: semantics.absolute_value(v, point.dtypes[n])
                    for n, v in row_major_args.items()
                    if torch.is_tensor(v)
                }
                compare_kwargs["magnitude_outputs"] = contract.mutable_tensors_by_name(
                    reference_fn(point, row_major_args | magnitudes, semantics.to_float32)
                )
            if request.has_error_bound("baseline_relative"):
                baseline = contract.baseline()
                if baseline is None:
                    raise ValueError(f"{contract.contract_id}: a baseline_relative bound needs a baseline")
                call = request.assemble_call(row_major_args, device)
                compare_kwargs["baseline_outputs"] = contract.call_mutable_tensors(baseline(**call), call)
            compared = functools.partial(semantics.compare_outputs, request.schema_json, **compare_kwargs)
            if not is_reference_comparable(mutable_dtypes, compare_kwargs["reference_outputs"], compared):
                detail = "the reference rounded to the output dtypes is not finite or fails compare"
                if first.input_set == "workload":
                    self._fail(first, FindingCode.SCHEMA_INVALID, detail)
                else:
                    self._fail(first, FindingCode.PLATFORM_FAILURE, f"internal: {detail}")
                continue
            for step in (first, *rest):
                actual_mutable = {o: self.outputs[step.output_key(o)] for o in contract.mutable_tensors}
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
