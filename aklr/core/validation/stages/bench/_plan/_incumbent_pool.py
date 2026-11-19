###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A Candidate's incumbent pool: the base commit's schemas of its name, seen as dispatch serves them on a target."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import WorkloadSpec as Workload
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.validation.stages.bench._plan._opponent import IncumbentOpponent
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.validation_scope import ValidationScope
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["IncumbentPool"]


@dataclass(frozen=True)
class IncumbentPool:
    schema: Schema  # the Candidate's
    schemas: dict[str, Schema]  # the base's, of this contract version, by base path
    target: DeviceTarget
    min_speedup: float  # AKLR_MIN_SPEEDUP, the bar at which a base schema serves
    baseline: str | None  # the Baseline's name a base schema's speedups are looked up by

    @classmethod
    def read(
        cls,
        scope: ValidationScope,
        path: str,
        schema: Schema,
        target: DeviceTarget,
        min_speedup: float,
        baseline: str | None,
    ) -> IncumbentPool:
        """The base files named as schema under the impls directory of path's op; none without a base. Raises
        FindingRaised, the platform's, at a base file that is not JSON; one of another contract version (it does
        not parse, or a workload of it is no Point of the Candidate's contract) is skipped."""
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
                    continue  # another contract version: never an Incumbent
                schemas[base_path] = base
        return cls(schema, schemas, target, min_speedup, baseline)

    def _served(self, base: Schema, workload: Workload) -> Workload | None:
        """base's workload at workload's Point, the one of its own key first; None when base has no kernel or no
        such workload."""
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
        """Whether base's kernel runs on the target, with the toolchains here."""
        return (
            DeviceTarget.matches_target(self.target, base.kernel.targets)
            and unmet_requirement(base.kernel.requires) is None
        )

    def defends(self, workload: Workload) -> bool:
        """Whether the base holds the Candidate's own kernel at workload's Point: it need not qualify, only run
        on the target."""
        return any(
            base.kernel.same_kernel(self.schema.kernel) and self._runs_here(base)
            for base in self.schemas.values()
            if self._served(base, workload) is not None
        )

    def incumbent(self, workload: Workload) -> IncumbentOpponent | None:
        """Another kernel the base serves workload's Point by: the higher stored speedup first, then the kernel
        hash and the path. A stored speedup is carried only from a base workload of workload's own key: one of
        other inputs is not."""
        own = self.schema.kernel.content_hash
        key = WorkloadKey.from_workload(workload)
        ranked = []
        for path, base in self.schemas.items():
            served = self._served(base, workload)
            if served is None or base.kernel.content_hash == own:
                continue
            # dispatch's rule: a kernel must beat the defined baseline, if there is one
            min_speedup = self.min_speedup if self.baseline else 0
            if not (self._runs_here(base) and served.qualifies(min_speedup)):
                continue
            speedup = served.speedup if WorkloadKey.from_workload(served) == key else None
            rank = (-(speedup or 0.0), base.kernel.content_hash, path)
            ranked.append((rank, IncumbentOpponent(self.baseline, path, base, speedup)))
        return min(ranked, key=lambda item: item[0], default=(None, None))[1]
