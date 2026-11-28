###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import WorkloadSpec as Workload
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.validation.validators.bench._plan._opponent import IncumbentOpponent
from aklr.core.validation.validators.report._finding import FindingCode, FindingRaised
from aklr.core.validation.validators.validation_scope import ValidationScope
from aklr.core.validation.validators.workload_key import WorkloadKey

__all__ = ["IncumbentPool"]


@dataclass(frozen=True)
class IncumbentPool:
    schema: Schema
    schemas: dict[str, Schema]
    target: DeviceTarget
    min_speedup: float
    baseline: str

    @classmethod
    def read(
        cls,
        scope: ValidationScope,
        path: str,
        schema: Schema,
        target: DeviceTarget,
        min_speedup: float,
        baseline: str,
    ) -> IncumbentPool:
        registry = ContractRegistry.default()
        contract = registry.get(schema.contract_id)
        schemas = {}
        if scope.base_sha is not None:
            for base_path in scope.paths_at_base(str(PurePosixPath(path).parents[2])):
                is_schema = ValidationScope.path_kind(base_path) == "schema_data"
                if not is_schema or PurePosixPath(base_path).parent.name != schema.name:
                    continue
                text = scope.text_at_base(base_path)
                try:
                    json.loads(text)
                except ValueError as error:
                    detail = f"the base file {base_path} is not JSON: {error}"
                    raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
                try:
                    base = registry.parse_schema(text.encode(), base_path)
                    for workload in base.workloads:
                        contract.check_point(base.point_with_contract_id(workload.axes, contract.contract_id))
                except ValueError:
                    continue
                schemas[base_path] = base
        return cls(schema, schemas, target, min_speedup, baseline)

    def _served(self, base: Schema, workload: Workload) -> Workload | None:
        if not base.kernel.sources:
            return None
        contract_id = self.schema.contract_id
        point_key = self.schema.point_with_contract_id(workload.axes, contract_id).key()
        key = WorkloadKey.from_workload(workload)
        return min(
            (
                w
                for w in base.workloads
                if base.point_with_contract_id(w.axes, contract_id).key() == point_key
            ),
            key=lambda w: WorkloadKey.from_workload(w) != key,
            default=None,
        )

    def _runs_here(self, base: Schema) -> bool:
        return (
            DeviceTarget.matches_target(self.target, base.kernel.targets)
            and unmet_requirement(base.kernel.requires) is None
        )

    def defends(self, workload: Workload) -> bool:
        return any(
            base.kernel.same_kernel(self.schema.kernel) and self._runs_here(base)
            for base in self.schemas.values()
            if self._served(base, workload) is not None
        )

    def incumbent(self, workload: Workload) -> IncumbentOpponent | None:
        own = self.schema.kernel.content_hash
        key = WorkloadKey.from_workload(workload)
        ranked = []
        for path, base in self.schemas.items():
            served = self._served(base, workload)
            if served is None or base.kernel.content_hash == own:
                continue
            if not (self._runs_here(base) and served.qualifies(self.min_speedup)):
                continue
            is_same_call = WorkloadKey.from_workload(served) == key and served.scalars == workload.scalars
            speedup = served.speedup if is_same_call else None
            rank = (speedup or 0.0, base.kernel.content_hash, path)
            ranked.append((rank, IncumbentOpponent(self.baseline, path, base, speedup)))
        return max(ranked, key=lambda item: item[0], default=(None, None))[1]
