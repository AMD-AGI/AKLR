###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's plan: for each benched schema with a kernel, what each workload is timed against and the threshold
it must clear."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.runtime.dispatch import OpVariants
from aklr.core.schema import Schema
from aklr.core.validation.stages.bench._plan._incumbent_pool import IncumbentPool
from aklr.core.validation.stages.bench._plan._opponent import BaselineOpponent, Opponent, Unopposed
from aklr.core.validation.stages.bench._plan._threshold import Threshold
from aklr.core.validation.stages.report._finding import Finding, FindingCode, FindingRaised
from aklr.core.validation.stages.validation_scope import ValidationScope
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["SchemaPlan", "WorkloadPlan"]


@dataclass(frozen=True)
class WorkloadPlan:
    """A workload's plan: its key, what its Candidate is timed against, and the threshold it must clear where it
    is timed."""

    key: WorkloadKey
    opponent: Opponent
    threshold: Threshold


@dataclass(frozen=True)
class SchemaPlan:
    path: str
    schema: Schema
    workloads: tuple[WorkloadPlan, ...] = ()
    findings: tuple[Finding, ...] = ()  # the platform's: the schema is not measured

    @classmethod
    def for_scope(cls, scope: ValidationScope, target: DeviceTarget, min_speedup: float) -> list[SchemaPlan]:
        """One plan per bench path whose schema has a kernel. min_speedup: AKLR_MIN_SPEEDUP, the bar at which a
        base schema serves, and so is an Incumbent."""
        import aklr.seam  # noqa: F401 -- declares the seam adapters
        from aklr.core.runtime.seam import SeamRegistry

        is_released = scope.base_sha is not None and not scope.is_first_release
        plans = []
        for absolute in scope.bench_paths:
            text = Path(absolute).read_text(encoding="utf-8")
            schema = OpVariants.load(text)
            if schema.kernel is None:
                continue
            path = scope.relative_path(absolute)
            baseline = None
            if schema.standard is not None and schema.standard.baseline is not None:
                # the name dispatch looks the speedups up by: the backend of the op's seam adapters
                baseline = next((a.backend for a in SeamRegistry.default().adapters(schema.op_type)), None)
            try:
                if (unmet := unmet_requirement(schema.kernel.build.requires)) is not None:
                    raise FindingRaised(FindingCode.PLATFORM_FAULT, f"kernel.build.requires: {unmet}")
                if not schema.kernel.build.runs_on(target):
                    targets = ", ".join(schema.kernel.build.target_hardware)
                    raise FindingRaised(
                        FindingCode.PLATFORM_FAULT, f"the kernel targets {targets}, not this {target}"
                    )
                pool = IncumbentPool.read(scope, path, schema, target, min_speedup, baseline)
            except FindingRaised as failure:
                finding = Finding(code=failure.code, schema_path=path, workload=None, detail=failure.detail)
                plans.append(cls(path=path, schema=schema, findings=(finding,)))
                continue
            plans.append(cls.from_schema(path, pool, is_released))
        return plans

    @classmethod
    def from_schema(cls, path: str, pool: IncumbentPool, is_released: bool) -> SchemaPlan:
        """Each workload's opponent and threshold: the one place that decides what a Candidate is timed against
        and the bar it must clear.

        pool: the Candidate's schema and what the base serves in its place; is_released: the base commit holds
        schema data, without which a Candidate needs only Retain over the Baseline."""
        workloads = []
        for workload in pool.schema.workloads:
            defends = pool.defends(workload)
            incumbent = None if defends else pool.incumbent(workload)
            opponent = incumbent or (BaselineOpponent(pool.baseline) if pool.baseline else Unopposed())
            threshold = Threshold.ENTRY if incumbent or (is_released and not defends) else Threshold.RETAIN
            key = WorkloadKey.from_workload(workload)
            workloads.append(WorkloadPlan(key=key, opponent=opponent, threshold=threshold))
        return cls(path=path, schema=pool.schema, workloads=tuple(workloads))
