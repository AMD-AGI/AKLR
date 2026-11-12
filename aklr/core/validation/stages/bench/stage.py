###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench stage: checks each benched kernel against its schema's Reference and times it."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, ClassVar

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.validation._execution import Executor, ResourceRequest
from aklr.core.validation.stages.bench._admission._plan import BenchPlanner, SchemaPlan
from aklr.core.validation.stages.bench._admission._verdict import SchemaVerdict
from aklr.core.validation.stages.bench._batch_schedule import BatchSchedule, CostliestFirst, WidestFirst
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.bench._measurement._workload_result import WorkloadResult
from aklr.core.validation.stages.bench._report import BenchReport
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
    ) -> None:
        """isolate: op types whose tasks each run in a fresh process rather than a reused one."""
        self.devices = tuple(devices)
        self.schedule = BatchSchedule(isolate=isolate, passes=(CostliestFirst(), WidestFirst()))
        self.profile = MeasurementProfile(num_trials=num_trials)

    def plan(self, scope: ValidationScope, target: DeviceTarget) -> list[SchemaPlan]:
        """One plan per bench path whose schema has a kernel."""
        return BenchPlanner(RuntimeEnv().min_speedup).plan(scope, target)

    @staticmethod
    def _join_process_group() -> None:
        """Each executor process's initializer: a runner's ranks, a single one too, join one process group,
        rendezvousing in its directory; gloo carries their agreements, RCCL the collectives of a group timing."""
        import torch
        import torch.distributed as dist

        size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{os.path.abspath('rendezvous')}",
            rank=int(os.environ["RANK"]),
            world_size=size,
            # a single rank runs no GPU collective: binding its device would start RCCL for nothing
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
        batches = self.schedule.batches(plans, self.profile)
        results: dict[str, list[WorkloadResult]] = {p.path: [] for p in plans}
        if batches:
            with Executor.for_devices(
                self.devices,
                runner="reuse",
                initializer=self._join_process_group,
                startup_timeout_secs=self.schedule.startup_timeout_secs,
            ) as executor:
                futures = {
                    executor.submit(
                        batch.run,
                        timeout_secs=self.schedule.timeout(batch),
                        resource_request=ResourceRequest(count=batch.execution_policy.ranks_per_task),
                        runner=self.schedule.runner(batch),
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    batch_results = batch.results(future)
                    results[batch.schema_path] += batch_results
                    self.progress.write(batch.summary(batch_results))
        verdicts = [
            SchemaVerdict.judge(p, results[p.path], self.profile, scope.head_sha, target) for p in plans
        ]
        return BenchReport.from_verdicts(self.name, verdicts, self.profile)
