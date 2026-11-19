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

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import unmet_requirement
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
    skipped: str | None = (
        None  # the kernel.requires this environment misses: not benched, as dispatch skips it
    )
    is_timed: bool = True  # False: its workloads are checked only, and nothing is written back

    @classmethod
    def for_scope(
        cls, scope: ValidationScope, target: DeviceTarget, min_speedup: float, timed: bool = True
    ) -> list[SchemaPlan]:
        """One plan per bench path whose schema has a kernel. min_speedup: AKLR_MIN_SPEEDUP, the bar at which a
        base schema serves, and so is an Incumbent. Not timed: the kernels are built and checked only."""
        registry = ContractRegistry.default()
        is_released = scope.base_sha is not None and not scope.is_first_release
        plans = []
        for absolute in scope.bench_paths:
            path = scope.relative_path(absolute)
            schema = registry.parse_schema(Path(absolute).read_bytes(), path)
            if not schema.kernel.sources:  # a definition: no kernel to bench
                continue
            # the contract's baseline, by its dotted name: what a stored speedup is over
            fn = registry.get(schema.contract_id).baseline()
            baseline = None if fn is None else f"{fn.__module__}.{fn.__qualname__}"
            if (unmet := unmet_requirement(schema.kernel.requires)) is not None:
                plans.append(cls(path=path, schema=schema, skipped=unmet))
                continue
            try:
                if not DeviceTarget.matches_target(target, schema.kernel.targets):
                    targets = ", ".join(map(str, schema.kernel.targets))
                    raise FindingRaised(
                        FindingCode.PLATFORM_FAULT, f"the kernel targets {targets}, not this {target}"
                    )
                if timed:
                    pool = IncumbentPool.read(scope, path, schema, target, min_speedup, baseline)
                    plans.append(cls.from_schema(path, pool, is_released))
                else:
                    plans.append(cls.untimed(path, schema))
            except FindingRaised as failure:
                finding = Finding(code=failure.code, schema_path=path, workload=None, detail=failure.detail)
                plans.append(cls(path=path, schema=schema, findings=(finding,)))
        return plans

    @classmethod
    def untimed(cls, path: str, schema: Schema) -> SchemaPlan:
        """Each workload against nothing, but those recorded failed: dispatch never serves one, so there is
        nothing to check."""
        workloads = tuple(
            WorkloadPlan(key=WorkloadKey.from_workload(w), opponent=Unopposed(), threshold=Threshold.RETAIN)
            for w in schema.workloads
            if w.accuracy != "failed"
        )
        return cls(path=path, schema=schema, workloads=workloads, is_timed=False)

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
            # a speedup over an Incumbent is carried to the Baseline by its stored one: without it the
            # Baseline itself is timed, as a workload written back passed has a speedup over it
            if incumbent is not None and pool.baseline and incumbent.stored_speedup is None:
                incumbent = None
            opponent = incumbent or (BaselineOpponent(pool.baseline) if pool.baseline else Unopposed())
            threshold = Threshold.ENTRY if incumbent or (is_released and not defends) else Threshold.RETAIN
            key = WorkloadKey.from_workload(workload)
            workloads.append(WorkloadPlan(key=key, opponent=opponent, threshold=threshold))
        return cls(path=path, schema=pool.schema, workloads=tuple(workloads))
