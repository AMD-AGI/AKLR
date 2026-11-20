###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


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
        self.max_workloads = max_workloads
        self.isolate = frozenset(isolate)
        self.build_timeout_secs = build_timeout_secs
        self.workload_timeout_secs = workload_timeout_secs
        self.startup_timeout_secs = startup_timeout_secs

    def tasks(self, plans: Sequence[SchemaPlan], profile: MeasurementProfile) -> list[BenchTask]:
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
        isolated = task.bench_type.isolate(task.schema) or task.schema.op_type in self.isolate
        return "disposable" if isolated else "reuse"

    def timeout(self, task: BenchTask) -> float:
        return self.build_timeout_secs + self.workload_timeout_secs * len(task.workloads)
