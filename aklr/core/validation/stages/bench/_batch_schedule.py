###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's schedule: its tasks' batches, where they run, and the passes that reorder them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence

from aklr.core.validation.stages.bench._admission._plan import SchemaPlan
from aklr.core.validation.stages.bench._execution_policy import ExecutionPolicy
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.bench._task._batch import Batch

__all__ = ["CostliestFirst", "BatchSchedule", "SchedulePass", "WidestFirst"]


class SchedulePass(ABC):
    @abstractmethod
    def apply(self, batches: list[Batch]) -> list[Batch]:
        """The batches reordered, split, merged or replaced; every workload still in exactly one batch of its
        schema, with the same profile."""


class CostliestFirst(SchedulePass):
    """The costliest batches start first: the shortest total time when batches outnumber devices."""

    def apply(self, batches: list[Batch]) -> list[Batch]:
        return sorted(batches, key=lambda b: b.cost, reverse=True)


class WidestFirst(SchedulePass):
    """The batches leasing the most devices start first: a wide batch waits for all its devices free at once,
    which never comes while narrow batches keep starting. The sort is stable: equal widths keep their order."""

    def apply(self, batches: list[Batch]) -> list[Batch]:
        return sorted(batches, key=lambda b: b.execution_policy.ranks_per_task, reverse=True)


class BatchSchedule:
    def __init__(
        self,
        *,
        batch_size: int = 16,
        isolate: Iterable[str] = (),
        build_timeout_secs: float = 900.0,
        workload_timeout_secs: float = 300.0,
        startup_timeout_secs: float = 300.0,
        passes: Sequence[SchedulePass] = (),
    ) -> None:
        """batch_size: workloads per batch at most; isolate: op types whose tasks each run in a fresh process;
        startup_timeout_secs: a runner's start, its process group included; one that overruns takes its devices
        out; passes: applied in order to the even split."""
        self.batch_size = batch_size
        self.isolate = frozenset(isolate)
        self.build_timeout_secs = build_timeout_secs
        self.workload_timeout_secs = workload_timeout_secs
        self.startup_timeout_secs = startup_timeout_secs
        self.passes = tuple(passes)

    def batches(self, plans: Sequence[SchemaPlan], profile: MeasurementProfile) -> list[Batch]:
        """Every plan's workloads in even batches of at most batch_size, each building the Candidate and its
        Incumbents once; in the order they start."""
        batches = []
        for plan in plans:
            n = len(plan.workloads)
            count = -(-n // self.batch_size)
            execution_policy = ExecutionPolicy.for_schema(plan.schema)
            batches += [
                Batch(
                    schema_path=plan.path,
                    schema_text=plan.schema_text,
                    workloads=plan.workloads[index * n // count : (index + 1) * n // count],
                    profile=profile,
                    execution_policy=execution_policy,
                )
                for index in range(count)
            ]
        planned = Counter((id(w), plan.path) for plan in plans for w in plan.workloads)
        for schedule_pass in self.passes:
            batches = schedule_pass.apply(batches)
            placed = Counter((id(w), b.schema_path) for b in batches for w in b.workloads)
            if placed != planned or any(b.profile is not profile for b in batches):
                raise ValueError(
                    f"{type(schedule_pass).__name__} lost, repeated or moved a workload, or changed the profile"
                )
        return batches

    def runner(self, batch: Batch) -> str:
        """The execution runner of its task: fresh processes when its policy or isolate asks for them."""
        isolated = batch.execution_policy.isolate or batch.schema.op_type in self.isolate
        return "disposable" if isolated else "reuse"

    def timeout(self, batch: Batch) -> float:
        """Its task's budget: building, then each workload's accuracy and timing."""
        return self.build_timeout_secs + self.workload_timeout_secs * len(batch.workloads)
