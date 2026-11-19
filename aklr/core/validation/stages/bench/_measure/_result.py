###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's result, as its task returns it: its findings and each timed trial's raw times."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from aklr.core.validation._frozen_model import frozen_dataclass
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.report._finding import Finding, FindingCode
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["Comparison", "WorkloadResult"]


@frozen_dataclass
class Comparison:
    """The Candidate timed against its opponent: raw times per trial, and their ratios' geometric mean."""

    candidate_ms: tuple[float, ...]
    opponent_ms: tuple[float, ...]
    ratio: float  # each trial's log ratio clamped: the gate's
    speedup: float  # unclamped, rounded: the one written back

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
    """Raw: the statistics a workload is judged by are never the task's to compute."""

    key: WorkloadKey
    findings: tuple[Finding, ...] = ()
    candidate_ms: tuple[float, ...] = ()  # one per timed trial
    opponent_ms: tuple[float, ...] = ()

    @classmethod
    def failed(cls, schema_path: str, key: WorkloadKey, code: FindingCode, detail: str) -> WorkloadResult:
        return cls(key, (Finding(code=code, schema_path=schema_path, workload=key, detail=detail),))

    def comparison(self, profile: MeasurementProfile) -> Comparison | None:
        """Its timings compared; None where it has none, or one side's are not the other's."""
        if not self.candidate_ms or len(self.candidate_ms) != len(self.opponent_ms):
            return None
        return Comparison.from_timings(profile, self.candidate_ms, self.opponent_ms)
