# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from aklr.core.contract import semantics
from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._bench._evaluation._comparison import (
    find_nondeterminism,
    is_reference_comparable,
)
from aklr.core.validation.stages._bench._evaluation._request import Step, TrialRequest
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
    float32 reference (float64 where float32 overflows), on their device; a group's, each
    rank's on its own."""

    request: TrialRequest
    outputs: Sequence[Mapping[str, torch.Tensor]]
    output_keys: Sequence[Sequence[str]]
    input_digests: Sequence[Mapping[str, str]]
    devices: Sequence[str]
    _failures: list[tuple[int, str | int, Finding]]

    def __init__(
        self,
        request: TrialRequest,
        outputs: Sequence[Mapping[str, torch.Tensor]],
        output_keys: Sequence[Sequence[str]],
        input_digests: Sequence[Mapping[str, str]],
        devices: Sequence[str],
    ) -> None:
        """By rank: ``outputs``, ``request.empty_outputs`` on ``devices``, of which the
        candidate filled ``output_keys`` from the input sets whose digests are
        ``input_digests``."""
        self.request = request
        self.outputs = outputs
        self.output_keys = output_keys
        self.input_digests = input_digests
        self.devices = devices
        self._failures = []

    def evaluate(self) -> ReferenceWorkloadResult:
        import torch

        request, devices = self.request, self.devices
        contract, point = request.contract, request.point
        ranks = range(len(devices))
        rank_at = [f"rank {r}: " if contract.ranks is not None else "" for r in ranks]

        def compose(step: Step) -> list[dict]:
            """Every rank's inputs, each on its device: a rank's reference may read its group's."""
            return [request.compose_inputs(step, devices[r], r) for r in ranks]

        steps = list(request.steps)
        composed: dict[str, list[dict]] = {}
        if step := next((s for s in steps if s.input_set == "workload"), None):
            try:
                composed[step.input_set] = compose(step)
            except ValueError as e:
                self._fail(step, FindingCode.SCHEMA_INVALID, str(e))
                steps.remove(step)
        n_present = len(steps)
        for rank in ranks:
            names = set(self.output_keys[rank])
            n_rank_present = 0
            while n_rank_present < len(steps) and names.issuperset(
                steps[n_rank_present].output_key(o) for o in contract.mutable_tensors
            ):
                n_rank_present += 1
            present_keys = {s.output_key(o) for s in steps[:n_rank_present] for o in contract.mutable_tensors}
            if extra := names - present_keys:
                detail = f"{rank_at[rank]}{sorted(extra)} are not outputs before a missing one"
                self._fail(None, FindingCode.CANDIDATE_CRASH, detail)
                return self._result(None, is_complete=False)
            n_present = min(n_present, n_rank_present)

        mutable_dtypes = {r: point.dtypes[r] for r in contract.mutable_tensors}
        worst = None
        passed_mutable: list[dict[str, dict[str, torch.Tensor]]] = [{} for _ in ranks]
        steps_by_inputs: dict[str, list[Step]] = {}
        for step in steps[:n_present]:
            steps_by_inputs.setdefault(step.input_set, []).append(step)
        for (
            first,
            *rest,
        ) in steps_by_inputs.values():
            group_args = composed.get(first.input_set) or compose(first)
            if request.has_error_bound("dot_product"):
                magnitudes = [
                    args
                    | {
                        n: semantics.absolute_value(v, point.dtypes[n])
                        for n, v in args.items()
                        if torch.is_tensor(v)
                    }
                    for args in group_args
                ]
            for rank in ranks:
                # the candidate's task drew the set on its own: other values make every compare moot
                if semantics.inputs_digest(group_args[rank]) != self.input_digests[rank].get(first.input_set):
                    detail = f"internal: {rank_at[rank]}the candidate's task drew other values of {first.input_set}"
                    self._fail(first, FindingCode.PLATFORM_FAILURE, detail)
                    continue
                for decode in (semantics.to_float32, semantics.to_float64):
                    compare_kwargs = {
                        "reference_outputs": contract.reference_outputs(point, group_args, rank, decode)
                    }
                    if request.has_error_bound("dot_product"):
                        compare_kwargs["magnitude_outputs"] = contract.reference_outputs(
                            point, magnitudes, rank, decode
                        )
                    # float32 overflows where the op's value can be finite (a sum that cancels): float64 then
                    if all(
                        t.isfinite().all() for outputs in compare_kwargs.values() for t in outputs.values()
                    ):
                        break
                if request.has_error_bound("baseline_relative"):  # no ranks: OpContract._check_ranks
                    baseline = contract.baseline()
                    if baseline is None:
                        raise ValueError(
                            f"{contract.contract_id}: a baseline_relative bound needs a baseline"
                        )
                    call = request.assemble_call(group_args[rank], devices[rank])
                    compare_kwargs["baseline_outputs"] = contract.call_mutable_tensors(baseline(**call), call)
                compared = functools.partial(semantics.compare_outputs, request.schema_json, **compare_kwargs)
                if not is_reference_comparable(mutable_dtypes, compare_kwargs["reference_outputs"], compared):
                    detail = f"{rank_at[rank]}the reference rounded to the output dtypes is not finite or fails compare"
                    if first.input_set == "workload":
                        self._fail(first, FindingCode.SCHEMA_INVALID, detail)
                    else:
                        self._fail(first, FindingCode.PLATFORM_FAILURE, f"internal: {detail}")
                    continue
                for step in (first, *rest):
                    actual_mutable = {
                        o: self.outputs[rank][step.output_key(o)] for o in contract.mutable_tensors
                    }
                    compare_results = compared(actual_mutable)
                    worst = max(
                        worst or 0.0,
                        *(r.violation_fraction for r in compare_results.values()),
                    )
                    if failed := {o: r for o, r in compare_results.items() if not r.is_passed}:
                        details = (f"{o}: {r.detail}" for o, r in failed.items())
                        self._fail(step, step.mismatch_code, rank_at[rank] + "; ".join(details))
                    elif step.name in ("numeric/0", "determinism/0"):
                        passed_mutable[rank][step.name] = actual_mutable
        for rank, passed in enumerate(passed_mutable):
            if len(passed) == 2:
                detail = find_nondeterminism(
                    request.schema_json, passed["numeric/0"], passed["determinism/0"]
                )
                if detail is not None:
                    step = next(s for s in steps if s.check == "determinism")
                    self._fail(step, FindingCode.NONDETERMINISTIC, rank_at[rank] + detail)
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
