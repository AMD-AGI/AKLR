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

from pydantic import TypeAdapter, ValidationError

from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch import DispatchKey
from aklr.core.schema import Schema, Workload
from aklr.core.validation.stages.bench._plan._opponent import IncumbentOpponent
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.validation_scope import ValidationScope
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["IncumbentPool"]


@dataclass(frozen=True)
class IncumbentPool:
    schema: Schema  # the Candidate's
    schemas: dict[str, Schema]  # the base's, of this schema format, by base path
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
        FindingRaised, the platform's, at a base file that is not JSON; one of another schema format is skipped."""
        schemas = {}
        if scope.base_sha is not None:
            for base_path in scope.paths_at_base(str(PurePosixPath(path).parents[2])):
                if PurePosixPath(base_path).parent.name != schema.name:
                    continue
                text = scope.text_at_base(base_path)
                try:
                    json.loads(text)
                except ValueError as error:
                    detail = f"the base file {base_path} is not JSON: {error}"
                    raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
                try:
                    schemas[base_path] = TypeAdapter(Schema).validate_json(text)
                except ValidationError:
                    continue  # another schema format: never an Incumbent
        return cls(schema, schemas, target, min_speedup, baseline)

    def _served(self, base: Schema, workload: Workload) -> Workload | None:
        """base's workload of workload's dispatch key, the one of its own key first; None when base has no kernel
        or no such workload."""
        if base.kernel is None:
            return None
        dispatch_key = DispatchKey.from_workload(base, workload)
        key = WorkloadKey.from_workload(workload)
        return min(
            (w for w in base.workloads if DispatchKey.from_workload(base, w) == dispatch_key),
            key=lambda w: WorkloadKey.from_workload(w) != key,
            default=None,
        )

    def defends(self, workload: Workload) -> bool:
        """Whether the base holds the Candidate's own kernel for workload's dispatch key: it need not qualify,
        only run on the target."""
        own = self.schema.kernel.hash()
        return any(
            base.kernel.hash() == own and base.kernel.build.runs_on(self.target)
            for base in self.schemas.values()
            if self._served(base, workload) is not None
        )

    def incumbent(self, workload: Workload) -> IncumbentOpponent | None:
        """Another kernel the base serves workload's dispatch key by: the higher stored speedup first, then the
        kernel hash and the path. A stored speedup is carried only from a base workload of workload's own key:
        one of other inputs is not."""
        own = self.schema.kernel.hash()
        key = WorkloadKey.from_workload(workload)
        ranked = []
        for path, base in self.schemas.items():
            served = self._served(base, workload)
            if served is None or base.kernel.hash() == own:
                continue
            min_speedup = self.min_speedup if self.baseline else 0
            if not base.serves(served, self.target, min_speedup, self.baseline or ""):
                continue
            evaluation = served.evaluation(self.target)
            stored = (
                evaluation.speedup.get(self.baseline) if evaluation is not None and self.baseline else None
            )
            speedup = None if stored is None or WorkloadKey.from_workload(served) != key else stored.value
            rank = (-(speedup or 0.0), base.kernel.hash(), path)
            ranked.append((rank, IncumbentOpponent(self.baseline, path, base, speedup)))
        return min(ranked, key=lambda item: item[0], default=(None, None))[1]
