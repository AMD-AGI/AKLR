# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._bench._evaluation._comparison import (
    find_nondeterminism,
)
from aklr.core.validation.stages._bench._evaluation._request import Step, TrialRequest
from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    import torch

__all__ = ["ReferenceWorkloadResult", "WorkloadEvaluation"]


class ReferenceWorkloadResult(FrozenModel):
    axes_text: str
    findings: tuple[Finding, ...]
    is_compared: bool  # the reference was compared with some step's outputs
    is_complete: bool


class WorkloadEvaluation:
    """One workload's outputs, as the candidate left them in the lent tensors, against the
    reference, on their device; a group's, each rank's on its own."""

    request: TrialRequest
    outputs: Sequence[Mapping[str, torch.Tensor]]
    output_keys: Sequence[Sequence[str]]
    devices: Sequence[str]
    _failures: list[tuple[int, str | int, Finding]]

    def __init__(
        self,
        request: TrialRequest,
        outputs: Sequence[Mapping[str, torch.Tensor]],
        output_keys: Sequence[Sequence[str]],
        devices: Sequence[str],
    ) -> None:
        """By rank: ``outputs``, ``request.empty_outputs`` on ``devices``, of which the
        candidate filled ``output_keys``."""
        self.request = request
        self.outputs = outputs
        self.output_keys = output_keys
        self.devices = devices
        self._failures = []

    def evaluate(self) -> ReferenceWorkloadResult:
        request, devices = self.request, self.devices
        contract = request.contract
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
                return self._result(False, is_complete=False)
            n_present = min(n_present, n_rank_present)

        reference = contract.reference
        is_compared = False
        passed_mutable: list[dict[str, dict[str, torch.Tensor]]] = [{} for _ in ranks]
        steps_by_inputs: dict[str, list[Step]] = {}
        for step in steps[:n_present]:
            steps_by_inputs.setdefault(step.input_set, []).append(step)
        for (
            first,
            *rest,
        ) in steps_by_inputs.values():
            group_args = composed.get(first.input_set) or compose(first)
            calls = [request.reference_call(args, devices[r]) for r, args in enumerate(group_args)]
            for rank in ranks:
                if contract.ranks is None:
                    reference_outputs = reference.reference(**calls[rank])
                else:
                    reference_outputs = reference.reference(calls, rank)
                compared = functools.partial(
                    reference.check, expected=reference_outputs, schema=request.schema
                )
                for step in (first, *rest):
                    actual_mutable = {
                        o: self.outputs[rank][step.output_key(o)] for o in contract.mutable_tensors
                    }
                    is_compared = True
                    if failed := compared(actual_mutable):
                        details = (f"{o}: {why}" for o, why in failed.items())
                        self._fail(step, step.mismatch_code, rank_at[rank] + "; ".join(details))
                    elif step.name in ("numeric/0", "determinism/0"):
                        passed_mutable[rank][step.name] = actual_mutable
        for rank, passed in enumerate(passed_mutable):
            if len(passed) == 2:
                # only a bitwise schema has a determinism step
                detail = find_nondeterminism(passed["numeric/0"], passed["determinism/0"])
                if detail is not None:
                    step = next(s for s in steps if s.check == "determinism")
                    self._fail(step, FindingCode.NONDETERMINISTIC, rank_at[rank] + detail)
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
