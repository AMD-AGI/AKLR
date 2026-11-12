###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A batch's run: one rank's built kernels, and the skeleton each workload runs through on every rank of its group."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from aklr.core.schema import Schema
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.bench._measurement._workload_result import (
    Comparison,
    WorkloadResult,
)
from aklr.core.validation.stages.bench._task._graph_timer import GraphTimer, TimingFault
from aklr.core.validation.stages.bench._workload_call._kernel_call import KernelCall
from aklr.core.validation.stages.bench._workload_call._output_check import OutputCheck
from aklr.core.validation.stages.bench._workload_call._reference_call import ReferenceCall
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.stages.bench._admission._opponent import Opponent
    from aklr.core.validation.stages.bench._admission._plan import WorkloadPlan

__all__ = ["BatchRun"]


class BatchRun:
    def __init__(
        self,
        schema_path: str,
        candidate: KernelCall,
        profile: MeasurementProfile,
        device: Any,
    ) -> None:
        self.schema_path = schema_path
        self.candidate = candidate
        self.profile = profile
        self.device = device
        self.incumbents: dict[str, KernelCall | Exception] = {}  # built once, by base path

    def result(self, plan: WorkloadPlan) -> WorkloadResult:
        """The workload checked on each trial, and each accurate trial timed against its opponent."""
        try:
            timed = [
                comparison
                for trial, call in enumerate(self.check(plan))
                if (comparison := self.time(plan.opponent, call, plan.key, trial)) is not None
            ]
        except FindingRaised as failure:
            return WorkloadResult.failed(self.schema_path, plan.key, failure.code, failure.detail)
        if not timed:
            return WorkloadResult(key=plan.key)
        comparison = Comparison.from_timings(
            self.profile,
            timed[0].versus,
            timed[0].versus_name,
            [ms for c in timed for ms in c.candidate_ms],
            [ms for c in timed for ms in c.opponent_ms],
        )
        return WorkloadResult(key=plan.key, comparison=comparison)

    def check(self, plan: WorkloadPlan) -> Iterator[dict[str, Any]]:
        """The Candidate against the Reference on num_trials input sets, each trial's call yielded once it passes
        on every rank; raises FindingRaised at a failing trial, the same on every rank.

        Each rank draws every rank's input set by the workload's sampler, calls the Candidate on its own, and
        checks it against its own Reference, which reads the group's input sets when it declares group_calls."""
        import torch.distributed as dist

        rank = dist.get_rank()
        schema = self.candidate.schema
        with FindingCode.PLATFORM_FAULT.failing_every_rank("the schema's Reference or check"):
            reference = ReferenceCall(schema)
            output_check = OutputCheck(schema, plan.key.axes)
        for trial in range(self.profile.num_trials):
            # the platform's or the schema's, never the Candidate's
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: drawing the input sets"):
                input_sets = plan.sampler.draw_group(trial, dist.get_world_size(), self.device)
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: the Reference"):
                expected = reference.outputs(input_sets, rank)
            call = input_sets[rank]
            del input_sets  # the other ranks' inputs go once the Reference has read them
            with FindingCode.CANDIDATE_RAISED.failing_every_rank(f"trial {trial}"):
                output_check.verify(expected, self.candidate.bind(call, plan.key.axes)(), trial)
            yield call

    def incumbent(self, path: str, text: str) -> KernelCall:
        """The Incumbent at path, built once; one that does not build is the platform's: it passed before."""
        if path not in self.incumbents:
            try:
                self.incumbents[path] = KernelCall.build(TypeAdapter(Schema).validate_json(text))
            except Exception as error:  # noqa: BLE001
                self.incumbents[path] = error
        built = self.incumbents[path]
        if isinstance(built, Exception):
            raise FindingRaised(FindingCode.PLATFORM_FAULT, f"the Incumbent {path} does not build: {built!r}")
        return built

    def time(
        self, opponent: Opponent, call: dict[str, Any], key: WorkloadKey, trial: int
    ) -> Comparison | None:
        """The Candidate and its opponent timed once each on a trial's call, in a seeded shuffled order; None when
        nothing is timed. Binding the opponent and its first call are agreed on by every rank first: a group
        timing's warm-up would leave the other ranks waiting on one that raised there."""
        versus, name = opponent.label, opponent.name
        # a Baseline's or an Incumbent's, never the Candidate's
        with FindingCode.PLATFORM_FAULT.failing_every_rank(f"binding the {versus} {name}"):
            bound = opponent.bind(self, call, key)
        if bound is None:
            return None
        with FindingCode.PLATFORM_FAULT.failing_every_rank(f"the {versus} {name}"):
            bound()
        candidate = self.candidate.bind(call, key.axes)
        try:
            seed = key.timing_seed(self.candidate.schema.name, trial)
            candidate_ms, opponent_ms = GraphTimer(self.profile.graph_budget_ms).time(candidate, bound, seed)
        except TimingFault as fault:
            code = (
                FindingCode.CANDIDATE_RAISED
                if fault.participant == "candidate"
                else FindingCode.PLATFORM_FAULT
            )
            raise FindingRaised(code, f"timing against the {versus} {name}: {fault}") from fault
        return Comparison.from_timings(self.profile, versus, name, [candidate_ms], [opponent_ms])
