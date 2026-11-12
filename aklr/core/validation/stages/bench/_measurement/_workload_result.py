###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's result, as its task returns it: its findings and its timed comparison."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.report._finding import Finding, FindingCode
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["Comparison", "WorkloadResult"]


@frozen_dataclass
class Comparison:
    """The Candidate timed against one opponent: raw times per trial, and their ratios' geometric mean."""

    versus: str  # what the Candidate is timed against: its opponent's label
    versus_name: str  # the Baseline's backend, or the Incumbent's schema path
    candidate_ms: tuple[float, ...]
    opponent_ms: tuple[float, ...]
    ratio: float  # each trial's log ratio clamped: the gate's
    speedup: float  # unclamped, rounded: the one written back

    @classmethod
    def from_timings(
        cls,
        profile: MeasurementProfile,
        versus: str,
        versus_name: str,
        candidate_ms: Sequence[float],
        opponent_ms: Sequence[float],
    ) -> Comparison:
        logs = [math.log(o / c) for c, o in zip(candidate_ms, opponent_ms, strict=True)]
        bound = profile.max_log_ratio
        return cls(
            versus=versus,
            versus_name=versus_name,
            candidate_ms=tuple(candidate_ms),
            opponent_ms=tuple(opponent_ms),
            ratio=math.exp(statistics.fmean(min(max(x, -bound), bound) for x in logs)),
            speedup=max(round(math.exp(statistics.fmean(logs)), 3), 0.001),
        )


@frozen_dataclass
class WorkloadResult:
    key: WorkloadKey
    findings: tuple[Finding, ...] = ()
    comparison: Comparison | None = None

    @classmethod
    def failed(cls, schema_path: str, key: WorkloadKey, code: FindingCode, detail: str) -> WorkloadResult:
        return cls(key, (Finding(code=code, schema_path=schema_path, workload=key, detail=detail),))
