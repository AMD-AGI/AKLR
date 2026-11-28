###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from aklr.core.common.frozen_model import frozen_dataclass
from aklr.core.validation.validators.bench._measure._profile import MeasurementProfile
from aklr.core.validation.validators.report._finding import Finding, FindingCode
from aklr.core.validation.validators.workload_key import WorkloadKey

__all__ = ["Comparison", "WorkloadResult"]


@frozen_dataclass
class Comparison:
    candidate_ms: tuple[float, ...]
    opponent_ms: tuple[float, ...]
    ratio: float
    speedup: float

    @classmethod
    def from_timings(
        cls, profile: MeasurementProfile, candidate_ms: Sequence[float], opponent_ms: Sequence[float]
    ) -> Comparison:
        logs = [math.log(o / c) for c, o in zip(candidate_ms, opponent_ms, strict=True)]
        bound = profile.max_log_ratio
        return cls(
            candidate_ms=tuple(candidate_ms),
            opponent_ms=tuple(opponent_ms),
            ratio=math.exp(statistics.fmean(min(max(x, -bound), bound) for x in logs)),
            speedup=max(round(math.exp(statistics.fmean(logs)), 3), 0.001),
        )


@frozen_dataclass
class WorkloadResult:
    key: WorkloadKey
    findings: tuple[Finding, ...] = ()
    candidate_ms: tuple[float, ...] = ()
    opponent_ms: tuple[float, ...] = ()

    @classmethod
    def failed(cls, schema_path: str, key: WorkloadKey, code: FindingCode, detail: str) -> WorkloadResult:
        return cls(key, (Finding(code=code, schema_path=schema_path, workload=key, detail=detail),))

    def comparison(self, profile: MeasurementProfile) -> Comparison | None:
        if not self.candidate_ms or len(self.candidate_ms) != len(self.opponent_ms):
            return None
        return Comparison.from_timings(profile, self.candidate_ms, self.opponent_ms)
