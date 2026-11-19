###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""An op type's bench: how its tasks run and how each of their workloads is checked and timed."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from aklr.core.common import Registrable
from aklr.core.validation.stages.bench._measure._built_kernel import BuiltKernel
from aklr.core.validation.stages.bench._measure._input_sampler import InputSampler
from aklr.core.validation.stages.bench._measure._output_check import OutputCheck
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.bench._measure._result import WorkloadResult
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract as Schema
    from aklr.core.validation.stages.bench._plan._plan import WorkloadPlan

__all__ = ["OpBench"]


class OpBench(Registrable):
    """The default bench: one rank, a reused process, no setup, inputs drawn by the schema's distributions and timed in
    CUDA graphs. An op type that differs registers a subclass in _ops/<op_type>.py and overrides only what
    differs.

    The class says how the op type's tasks run, from the schema alone: ranks, isolate, setup. An instance is one
    task's run on one rank, once the Candidate is built: measure, and the steps it is made of (trials, draws,
    time, clock), from coarse to fine."""

    @staticmethod
    def for_op(op_type: str) -> type[OpBench]:
        """The bench registered under op_type, else this default."""
        import importlib
        import pkgutil

        from aklr.core.validation.stages.bench import _ops

        for module in pkgutil.iter_modules(_ops.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{_ops.__name__}.{module.name}")
        try:
            return OpBench.resolve(op_type)
        except ValueError:  # no bench registered under this op type
            return OpBench

    @classmethod
    def ranks(cls, schema: Schema) -> int:
        """Ranks that run a task together, one device each."""
        return 1

    @classmethod
    def isolate(cls, schema: Schema) -> bool:
        """Whether each task runs in new processes, as --isolate makes it."""
        return cls.ranks(schema) > 1

    @classmethod
    def setup(cls, schema: Schema) -> None:
        """Prepare each rank's process before the build; idempotent unless isolate; no global numeric setting."""

    def __init__(
        self, schema_path: str, candidate: BuiltKernel, profile: MeasurementProfile, device: Any
    ) -> None:
        self.schema_path = schema_path
        self.candidate = candidate
        self.profile = profile
        self.device = device
        self.incumbents: dict[str, BuiltKernel | Exception] = {}  # built once, by base path

    def incumbent(self, path: str, schema: Schema) -> BuiltKernel:
        """The Incumbent at path, built once; one that does not build is the platform's: it passed before. Not a
        step to override: what an Incumbent opponent binds."""
        if path not in self.incumbents:
            try:
                self.incumbents[path] = BuiltKernel.build(self.candidate.schema, self.device, schema)
            except Exception as error:  # noqa: BLE001
                self.incumbents[path] = error
        built = self.incumbents[path]
        if isinstance(built, Exception):
            raise FindingRaised(FindingCode.PLATFORM_FAULT, f"the Incumbent {path} does not build: {built!r}")
        return built

    def measure(self, plan: WorkloadPlan) -> WorkloadResult:
        """A workload's result: checked on each trial, and each accurate trial's times against its opponent. Raises
        FindingRaised, the same on every rank, where a step fails."""
        timed = [
            pair
            for trial, inputs in enumerate(self.trials(plan))
            if (pair := self.time(plan, inputs, trial)) is not None
        ]
        return WorkloadResult(
            key=plan.key,
            candidate_ms=tuple(candidate_ms for candidate_ms, _ in timed),
            opponent_ms=tuple(opponent_ms for _, opponent_ms in timed),
        )

    def trials(self, plan: WorkloadPlan) -> Iterator[dict[str, Any]]:
        """The Candidate against the Reference on each trial's input sets, this rank's inputs yielded once the
        trial passes on every rank; raises FindingRaised at the first failing trial, the same on every rank.

        Each rank has every rank's input set, calls the Candidate on its own, and checks it against its own
        Reference, which reads the group's input sets when it declares group_calls."""
        import torch.distributed as dist

        rank = dist.get_rank()
        schema = self.candidate.schema
        with FindingCode.PLATFORM_FAULT.failing_every_rank("the schema's Reference or check"):
            output_check = OutputCheck(schema, plan.key.axes)
        draws = self.draws(plan)
        for trial in itertools.count():
            # the platform's or the schema's, never the Candidate's
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: drawing the input sets"):
                input_sets = next(draws, None)
            if input_sets is None:
                return
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: the Reference"):
                reference_outputs = output_check.reference(input_sets, rank)
            inputs = input_sets[rank]
            del input_sets  # the other ranks' inputs go once the Reference has read them
            with FindingCode.CANDIDATE_RAISED.failing_every_rank(f"trial {trial}"):
                output_check.verify(reference_outputs, self.candidate.bind(inputs, plan.key.axes)(), trial)
            yield inputs

    def draws(self, plan: WorkloadPlan) -> Iterator[list[dict[str, Any]]]:
        """Every trial's input sets, one trial at a time, one input set per rank: how many trials a workload has
        is what this yields. The same on every rank and every run, in the same order; lazy, so that one trial's
        inputs are on the device at a time. The default: num_trials draws by the schema's distributions."""
        import torch.distributed as dist

        sampler = InputSampler(self.candidate.schema, plan.key.axes)
        for trial in range(self.profile.num_trials):
            yield sampler.draw_group(trial, dist.get_world_size(), self.device)

    def time(self, plan: WorkloadPlan, inputs: dict[str, Any], trial: int) -> tuple[float, float] | None:
        """The Candidate's and its opponent's milliseconds per call on a trial's inputs, one clock reading each,
        in a seeded shuffled order; None when nothing is timed.

        Everything about a timing but reading the clock is here. Binding the opponent and its first call are
        agreed on by every rank first: a group timing's warm-up would leave the other ranks waiting on one that
        raised there. A reading whose call raised is that participant's on every rank: the Candidate's own, or
        the platform's for its opponent. A clock that fails otherwise, or reads no positive time, is read once
        more, then the platform's."""
        import torch.distributed as dist

        opponent = plan.opponent
        versus = f"the {opponent.label} {opponent.name}"
        # a Baseline's or an Incumbent's, never the Candidate's
        with FindingCode.PLATFORM_FAULT.failing_every_rank(f"binding {versus}"):
            bound = opponent.bind(self, inputs, plan.key)
        if bound is None:
            return None
        with FindingCode.PLATFORM_FAULT.failing_every_rank(versus):
            bound()
        participants = {"candidate": self.candidate.bind(inputs, plan.key.axes), "opponent": bound}
        order = list(participants)
        random.Random(plan.key.seed(self.candidate.schema.name, plan.key.axes, f"timing {trial}")).shuffle(
            order
        )
        timings: dict[str, float] = {}
        for participant in order:
            fn = participants[participant]
            raised: list[BaseException] = []

            def call(fn: Callable[[], object] = fn, raised: list[BaseException] = raised) -> None:
                try:
                    fn()
                except BaseException as error:
                    raised.append(error)
                    raise

            fault: BaseException = RuntimeError("no timing")
            for _ in range(2):
                try:
                    ms = self.clock(call)
                except Exception as error:  # noqa: BLE001 -- the participant's, else the clock's
                    own = f"{type(raised[0]).__name__}: {raised[0]}" if raised else None
                    messages: list[str | None] = [None] * dist.get_world_size()
                    dist.all_gather_object(messages, own)
                    for rank, message in enumerate(messages):
                        if message is not None:
                            if len(messages) > 1:
                                message = f"RuntimeError: rank {rank}: {message}"
                            code = (
                                FindingCode.CANDIDATE_RAISED
                                if participant == "candidate"
                                else FindingCode.PLATFORM_FAULT
                            )
                            detail = f"timing against {versus}: {participant}: {message}"
                            raise FindingRaised(code, detail) from error
                    fault = error
                    continue
                if math.isfinite(ms) and ms > 0:
                    timings[participant] = ms
                    break
                fault = RuntimeError(f"the clock read {ms} ms")
            else:
                detail = f"timing against {versus}: tracer: {type(fault).__name__}: {fault}"
                raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from fault
        return timings["candidate"], timings["opponent"]

    def clock(self, fn: Callable[[], object]) -> float:
        """fn's milliseconds per call: reading the time and nothing else. On a group, every rank returns the same
        reading. The default: fn's GPU time in a CUDA graph filling the profile's graph budget, the slowest
        rank's."""
        import torch.distributed as dist

        from aklr.core.utils.testing import bench_cudagraph

        return bench_cudagraph(fn, rep=self.profile.graph_budget_ms, group=dist.group.WORLD)
