###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from aklr.core.common import Registrable
from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.validators.bench._measure._built_kernel import BuiltKernel
from aklr.core.validation.validators.bench._measure._input_sampler import InputSampler
from aklr.core.validation.validators.bench._measure._output_check import OutputCheck
from aklr.core.validation.validators.bench._measure._profile import MeasurementProfile
from aklr.core.validation.validators.bench._measure._result import WorkloadResult
from aklr.core.validation.validators.report._finding import FindingCode, FindingRaised

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract as Schema
    from aklr.core.validation.validators.bench._plan._plan import WorkloadPlan

__all__ = ["OpBench"]


class OpBench(Registrable):
    @staticmethod
    def for_op(op_type: str) -> type[OpBench]:
        import importlib
        import pkgutil

        from aklr.core.validation.validators.bench import _custom

        for module in pkgutil.iter_modules(_custom.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{_custom.__name__}.{module.name}")
        try:
            return OpBench.resolve(op_type)
        except ValueError:
            return OpBench

    @classmethod
    def ranks(cls, schema: Schema) -> int:
        return 1

    @classmethod
    def isolate(cls, schema: Schema) -> bool:
        return cls.ranks(schema) > 1

    @classmethod
    def setup(cls, schema: Schema) -> None:
        ContractRegistry.default().get(schema.contract_id).baseline.setup()

    @classmethod
    def teardown(cls, schema: Schema) -> None:
        ContractRegistry.default().get(schema.contract_id).baseline.teardown()

    def __init__(
        self, schema_path: str, candidate: BuiltKernel, profile: MeasurementProfile, device: Any
    ) -> None:
        self.schema_path = schema_path
        self.candidate = candidate
        self.profile = profile
        self.device = device
        self.incumbents: dict[str, BuiltKernel | Exception] = {}

    def incumbent(self, path: str, schema: Schema) -> BuiltKernel:
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
        import torch.distributed as dist

        rank = dist.get_rank()
        schema = self.candidate.schema
        with FindingCode.PLATFORM_FAULT.failing_every_rank("the schema's Reference or check"):
            output_check = OutputCheck(schema, plan.key.axes)
        draws = self.draws(plan)
        for trial in itertools.count():
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: drawing the input sets"):
                input_sets = next(draws, None)
            if input_sets is None:
                return
            with FindingCode.PLATFORM_FAULT.failing_every_rank(f"trial {trial}: the Reference"):
                reference_outputs = output_check.reference(input_sets, rank)
            inputs = input_sets[rank]
            del input_sets
            with FindingCode.CANDIDATE_RAISED.failing_every_rank(f"trial {trial}"):
                output_check.verify(reference_outputs, self.candidate.bind(inputs, plan.key.axes)(), trial)
            yield inputs

    def draws(self, plan: WorkloadPlan) -> Iterator[list[dict[str, Any]]]:
        import torch.distributed as dist

        sampler = InputSampler(self.candidate.schema, plan.key.axes)
        for trial in range(self.profile.num_trials):
            yield sampler.draw_group(trial, dist.get_world_size(), self.device)

    def time(self, plan: WorkloadPlan, inputs: dict[str, Any], trial: int) -> tuple[float, float] | None:
        import torch.distributed as dist

        opponent = plan.opponent
        versus = f"the {opponent.label} {opponent.name}"
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
        return self.candidate.dispatcher.contract.baseline.profile(fn)
