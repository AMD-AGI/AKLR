###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.validation.validators.bench._plan._incumbent_pool import IncumbentPool
from aklr.core.validation.validators.bench._plan._opponent import BaselineOpponent, Opponent, Unopposed
from aklr.core.validation.validators.bench._plan._threshold import Threshold
from aklr.core.validation.validators.report._finding import Finding, FindingCode, FindingRaised
from aklr.core.validation.validators.validation_scope import ValidationScope
from aklr.core.validation.validators.workload_key import WorkloadKey

__all__ = ["SchemaPlan", "WorkloadPlan"]


@dataclass(frozen=True)
class WorkloadPlan:
    key: WorkloadKey
    opponent: Opponent
    threshold: Threshold


@dataclass(frozen=True)
class SchemaPlan:
    path: str
    schema: Schema | None
    workloads: tuple[WorkloadPlan, ...] = ()
    findings: tuple[Finding, ...] = ()
    skipped: str | None = None
    is_timed: bool = True

    @classmethod
    def for_scope(
        cls, scope: ValidationScope, target: DeviceTarget, min_speedup: float, timed: bool = True
    ) -> list[SchemaPlan]:
        registry = ContractRegistry.default()
        is_released = scope.base_sha is not None and not scope.is_first_release
        plans = []
        for absolute in scope.bench_paths:
            path = scope.relative_path(absolute)
            try:
                schema = registry.parse_schema(Path(absolute).read_bytes(), path)
            except ValueError as error:
                finding = Finding(
                    code=FindingCode.SCHEMA_INVALID, schema_path=path, workload=None, detail=str(error)
                )
                plans.append(cls(path=path, schema=None, findings=(finding,)))
                continue
            if not schema.kernel.sources:
                continue
            baseline_type = type(registry.get(schema.contract_id).baseline)
            baseline = f"{baseline_type.__module__}.{baseline_type.__qualname__}"
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
        workloads = tuple(
            WorkloadPlan(key=WorkloadKey.from_workload(w), opponent=Unopposed(), threshold=Threshold.RETAIN)
            for w in schema.workloads
            if w.accuracy != "failed"
        )
        return cls(path=path, schema=schema, workloads=workloads, is_timed=False)

    @classmethod
    def from_schema(cls, path: str, pool: IncumbentPool, is_released: bool) -> SchemaPlan:
        workloads = []
        for workload in pool.schema.workloads:
            defends = pool.defends(workload)
            incumbent = None if defends else pool.incumbent(workload)
            if incumbent is not None and incumbent.stored_speedup is None:
                incumbent = None
            opponent = incumbent or BaselineOpponent(pool.baseline)
            threshold = Threshold.ENTRY if incumbent or (is_released and not defends) else Threshold.RETAIN
            key = WorkloadKey.from_workload(workload)
            workloads.append(WorkloadPlan(key=key, opponent=opponent, threshold=threshold))
        return cls(path=path, schema=pool.schema, workloads=tuple(workloads))
