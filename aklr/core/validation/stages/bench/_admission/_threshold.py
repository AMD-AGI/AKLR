###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A speed threshold: the bar a timed workload's ratio must clear, and what failing it is called."""

from __future__ import annotations

from enum import StrEnum

from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.report._finding import FindingCode

__all__ = ["Threshold"]


class Threshold(StrEnum):
    ENTRY = "entry"  # to start serving: faster than what serves today
    RETAIN = "retain"  # to keep serving: a small slowdown over the Baseline allowed

    @property
    def failing_code(self) -> FindingCode:
        return FindingCode.NOT_FASTER if self == Threshold.ENTRY else FindingCode.REGRESSED

    def passes(self, ratio: float, profile: MeasurementProfile) -> bool:
        """Whether a ratio's lower bound clears this threshold, at profile's bars."""
        return ratio > (profile.entry if self == Threshold.ENTRY else profile.retain)
