###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, ClassVar

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.validation._execution import Executor, ResourceRequest
from aklr.core.validation.stages.bench._judge._report import BenchReport
from aklr.core.validation.stages.bench._judge._verdict import SchemaVerdict
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.bench._measure._result import WorkloadResult
from aklr.core.validation.stages.bench._measure._schedule import TaskSchedule
from aklr.core.validation.stages.bench._plan._plan import SchemaPlan
from aklr.core.validation.stages.report._finding import Finding, FindingCode
from aklr.core.validation.stages.stage import PipelineStage

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport
    from aklr.core.validation.stages.validation_scope import ValidationScope

__all__ = ["BenchStage"]


class BenchStage(PipelineStage):
    name: ClassVar[str] = "bench"
    report_type: ClassVar[type[BenchReport]] = BenchReport

    def __init__(
        self,
        devices: Sequence[str],
        *,
        isolate: Iterable[str] = (),
        num_trials: int = 3,
        timed: bool = True,
    ) -> None:
        self.devices = tuple(devices)
        self.timed = timed
        self.schedule = TaskSchedule(isolate=isolate)
        self.profile = MeasurementProfile(num_trials=num_trials)

    def plan(self, scope: ValidationScope, target: DeviceTarget) -> list[SchemaPlan]:
        return SchemaPlan.for_scope(scope, target, RuntimeEnv().min_speedup, self.timed)

    @staticmethod
    def _join_process_group() -> None:
        import torch
        import torch.distributed as dist

        size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{os.path.abspath('rendezvous')}",
            rank=int(os.environ["RANK"]),
            world_size=size,
            device_id=torch.device("cuda", torch.cuda.current_device()) if size > 1 else None,
        )

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> BenchReport:
        platform = earlier.platform
        target = DeviceTarget.from_names(platform.arch, platform.hardware)
        if not target.is_complete:
            detail = f"unknown device {platform.arch}/{platform.hardware}"
            return BenchReport.from_findings(
                self.name, [Finding.unlocated(FindingCode.PLATFORM_FAULT, detail)], profile=self.profile
            )
        plans = self.plan(scope, target)
        for plan in plans:
            if plan.skipped is not None:
                self.progress.write(f"skipped {plan.path}: kernel.build.requires: {plan.skipped}")
        tasks = self.schedule.tasks(plans, self.profile)
        results: dict[str, list[WorkloadResult]] = {p.path: [] for p in plans}
        if tasks:
            with Executor.for_devices(
                self.devices,
                runner="reuse",
                initializer=self._join_process_group,
                startup_timeout_secs=self.schedule.startup_timeout_secs,
            ) as executor:
                futures = {
                    executor.submit(
                        task.run,
                        timeout_secs=self.schedule.timeout(task),
                        resource_request=ResourceRequest(count=task.ranks),
                        runner=self.schedule.runner(task),
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    task_results = task.results(future)
                    results[task.schema_path] += task_results
                    self.progress.write(task.summary(task_results))
        verdicts = [
            SchemaVerdict.judge(p, results[p.path], self.profile, scope.head_sha, target) for p in plans
        ]
        return BenchReport.from_verdicts(self.name, verdicts, self.profile)
