###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's schedule: its tasks in the order they start, and where and how long each runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from aklr.core.validation.stages.bench._measure._op_bench import OpBench
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.bench._measure._task import BenchTask
from aklr.core.validation.stages.bench._plan._plan import SchemaPlan

__all__ = ["TaskSchedule"]


class TaskSchedule:
    def __init__(
        self,
        *,
        max_workloads: int = 16,
        isolate: Iterable[str] = (),
        build_timeout_secs: float = 900.0,
        workload_timeout_secs: float = 300.0,
        startup_timeout_secs: float = 300.0,
    ) -> None:
        """max_workloads: a task's at most; isolate: op types whose tasks each run in a fresh process;
        startup_timeout_secs: a runner's start, its process group included; one that overruns takes its devices
        out."""
        self.max_workloads = max_workloads
        self.isolate = frozenset(isolate)
        self.build_timeout_secs = build_timeout_secs
        self.workload_timeout_secs = workload_timeout_secs
        self.startup_timeout_secs = startup_timeout_secs

    def tasks(self, plans: Sequence[SchemaPlan], profile: MeasurementProfile) -> list[BenchTask]:
        """Every plan's workloads in even tasks of at most max_workloads, each building the Candidate and its
        Incumbents once; in the order they start.

        The tasks leasing the most devices start first: a wide task waits for all its devices free at once, which
        never comes while narrow ones keep starting. Among equally wide ones the costliest start first: the
        shortest total time when tasks outnumber devices."""
        tasks = []
        for plan in plans:
            n = len(plan.workloads)
            count = -(-n // self.max_workloads)
            bench_type = OpBench.for_op(plan.schema.op_type)
            tasks += [
                BenchTask(
                    schema_path=plan.path,
                    schema=plan.schema,
                    workloads=plan.workloads[index * n // count : (index + 1) * n // count],
                    profile=profile,
                    bench_type=bench_type,
                )
                for index in range(count)
            ]
        return sorted(tasks, key=lambda t: (t.ranks, t.cost), reverse=True)

    def runner(self, task: BenchTask) -> str:
        """The execution runner of task: fresh processes when its op's bench or isolate asks for them."""
        isolated = task.bench_type.isolate(task.schema) or task.schema.op_type in self.isolate
        return "disposable" if isolated else "reuse"

    def timeout(self, task: BenchTask) -> float:
        """Its budget: building, then each workload's accuracy and timing."""
        return self.build_timeout_secs + self.workload_timeout_secs * len(task.workloads)
