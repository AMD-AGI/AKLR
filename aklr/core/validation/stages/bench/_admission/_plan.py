###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's plan: for each benched schema with a kernel, what each workload is timed against, the Incumbent
among the base commit's schemas included."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath

from pydantic import TypeAdapter, ValidationError

from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch._dispatch_key import DispatchKeyFactory
from aklr.core.schema import Schema, Workload
from aklr.core.validation.stages.bench._admission._opponent import (
    BaselineOpponent,
    IncumbentOpponent,
    Opponent,
    Unopposed,
)
from aklr.core.validation.stages.bench._admission._threshold import Threshold
from aklr.core.validation.stages.bench._workload_call._input_sampler import InputSampler
from aklr.core.validation.stages.report._finding import Finding, FindingCode
from aklr.core.validation.stages.validation_scope import ValidationScope
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["BenchPlanner", "Incumbent", "IncumbentPool", "SchemaPlan", "WorkloadPlan"]


@dataclass(frozen=True)
class Incumbent:
    path: str  # its base path
    text: str
    is_own: bool  # it holds the Candidate's own kernel
    speedup: float | None  # its stored speedup over the Baseline on this target


@dataclass(frozen=True)
class IncumbentPool:
    texts: dict[str, str] = field(default_factory=dict)  # by base path
    schemas: dict[str, Schema] = field(default_factory=dict)  # those of this schema format, by base path
    finding: Finding | None = None  # a base file that is not JSON: the schema is not measured

    @classmethod
    def read(cls, scope: ValidationScope, path: str, name: str) -> IncumbentPool:
        """The base files named name under the impls directory of path's op."""
        if scope.base_sha is None:
            return cls()
        texts = {}
        for base_path in scope.paths_at_base(str(PurePosixPath(path).parents[2])):
            if PurePosixPath(base_path).parent.name != name:
                continue
            texts[base_path] = scope.text_at_base(base_path)
            try:
                json.loads(texts[base_path])
            except ValueError as error:
                detail = f"the base file {base_path} is not JSON: {error}"
                return cls(
                    finding=Finding(
                        code=FindingCode.PLATFORM_FAULT, schema_path=path, workload=None, detail=detail
                    )
                )
        schemas = {}
        for base_path, text in texts.items():
            try:
                schemas[base_path] = TypeAdapter(Schema).validate_json(text)
            except ValidationError:
                continue  # another schema format: never an Incumbent
        return cls(texts, schemas)

    def incumbent(
        self,
        workload: Workload,
        kernel_hash: str,
        target: DeviceTarget,
        baseline: str | None,
        min_speedup: float,
    ) -> Incumbent | None:
        """Among the base schemas serving workload's dispatch key that qualify on target or hold the Candidate's
        kernel: its own kernel first, then accuracy not failed, the higher stored speedup, the kernel hash. A
        stored speedup is carried only from a base workload of workload's own key: one of other inputs is not."""
        workload_key = WorkloadKey.from_workload(workload)
        ranked = []
        for path, base in self.schemas.items():
            builder = DispatchKeyFactory.specialize(base)
            dispatch_key = builder.build_from_workload(workload)
            served = min(
                (w for w in base.workloads if builder.build_from_workload(w) == dispatch_key),
                key=lambda w: WorkloadKey.from_workload(w) != workload_key,  # its own key's first
                default=None,
            )
            # WORKAROUND: copies SchemaCollection.filter_schemas's serve rule by hand; remove when Schema owns it
            if base.kernel is None or served is None or not base.kernel.build.runs_on(target):
                continue
            is_own = base.kernel.hash() == kernel_hash
            if not is_own and not served.qualifies(target, min_speedup if baseline else 0, baseline or ""):
                continue
            evaluation = served.evaluation(target)
            stored = evaluation.speedup.get(baseline) if evaluation is not None and baseline else None
            speedup = (
                None if stored is None or WorkloadKey.from_workload(served) != workload_key else stored.value
            )
            is_failed = evaluation is not None and evaluation.accuracy == "failed"
            rank = (not is_own, is_failed, -(speedup or 0.0), base.kernel.hash(), path)
            ranked.append((rank, Incumbent(path, self.texts[path], is_own, speedup)))
        return min(ranked, key=lambda item: item[0], default=(None, None))[1]


@dataclass(frozen=True)
class WorkloadPlan:
    """A workload's plan: its key, what its Candidate is timed against and how its input sets are made."""

    key: WorkloadKey
    opponent: Opponent
    sampler: InputSampler


@dataclass(frozen=True)
class SchemaPlan:
    path: str
    schema_text: str
    workloads: tuple[WorkloadPlan, ...] = ()
    findings: tuple[Finding, ...] = ()  # the platform's: the schema is not measured

    @cached_property
    def schema(self) -> Schema:
        return TypeAdapter(Schema).validate_json(self.schema_text)

    @classmethod
    def from_schema(
        cls,
        path: str,
        text: str,
        baseline: str | None,
        pool: IncumbentPool,
        target: DeviceTarget,
        min_speedup: float,
        is_released: bool,
    ) -> SchemaPlan:
        """Each workload's opponent: the one place that decides what a Candidate is timed against.

        baseline: the name of the op's Baseline, None without one; is_released: the base commit holds schema data,
        without which a Candidate needs only Retain over the Baseline."""
        plan = cls(path=path, schema_text=text)
        if pool.finding is not None:
            return plan.unmeasured(pool.finding)
        kernel_hash = plan.schema.kernel.hash()
        workloads = []
        for workload in plan.schema.workloads:
            incumbent = pool.incumbent(workload, kernel_hash, target, baseline, min_speedup)
            if incumbent is not None and not incumbent.is_own:
                opponent = IncumbentOpponent(
                    Threshold.ENTRY, incumbent.path, incumbent.text, incumbent.speedup, baseline
                )
            elif baseline is None:
                opponent = Unopposed()
            elif incumbent is not None or not is_released:
                opponent = BaselineOpponent(Threshold.RETAIN, baseline)
            else:
                opponent = BaselineOpponent(Threshold.ENTRY, baseline)
            key = WorkloadKey.from_workload(workload)
            sampler = InputSampler.for_workload(plan.schema, key)
            workloads.append(WorkloadPlan(key=key, opponent=opponent, sampler=sampler))
        return cls(path=path, schema_text=text, workloads=tuple(workloads))

    def unmeasured(self, finding: Finding) -> SchemaPlan:
        return SchemaPlan(path=self.path, schema_text=self.schema_text, findings=(finding,))


class BenchPlanner:
    def __init__(self, min_speedup: float) -> None:
        """min_speedup: AKLR_MIN_SPEEDUP, the bar at which a base schema serves, and so is an Incumbent."""
        self.min_speedup = min_speedup

    def plan(self, scope: ValidationScope, target: DeviceTarget) -> list[SchemaPlan]:
        import aklr.seam  # noqa: F401 -- declares the seam adapters
        from aklr.core.runtime.seam import SeamRegistry

        is_released = scope.base_sha is not None and not scope.is_first_release
        plans = []
        for absolute in scope.bench_paths:
            text = Path(absolute).read_text(encoding="utf-8")
            schema = TypeAdapter(Schema).validate_json(text)
            if schema.kernel is None:
                continue
            path = scope.relative_path(absolute)
            baseline = None
            if schema.standard is not None and schema.standard.baseline is not None:
                # the name dispatch looks the speedups up by: the backend of the op's seam adapters
                baseline = next((a.backend for a in SeamRegistry.default().adapters(schema.op_type)), None)
            if schema.kernel.build.runs_on(target):
                pool = IncumbentPool.read(scope, path, schema.name)
            else:
                targets = ", ".join(schema.kernel.build.target_hardware)
                detail = f"the kernel targets {targets}, not this {target}"
                pool = IncumbentPool(
                    finding=Finding(
                        code=FindingCode.PLATFORM_FAULT, schema_path=path, workload=None, detail=detail
                    )
                )
            plans.append(
                SchemaPlan.from_schema(path, text, baseline, pool, target, self.min_speedup, is_released)
            )
        return plans
