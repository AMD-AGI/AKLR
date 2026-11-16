###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A speed threshold: the bar a timed workload's ratio must clear, and what failing it is called."""

from __future__ import annotations

from enum import StrEnum

from aklr.core.platform.env import DEFAULT_MIN_SPEEDUP
from aklr.core.validation.stages.report._finding import FindingCode

__all__ = ["Threshold"]


class Threshold(StrEnum):
    bar: float
    failing_code: FindingCode

    def __new__(cls, value: str, bar: float, failing_code: FindingCode) -> Threshold:
        member = str.__new__(cls, value)
        member._value_ = value
        member.bar = bar
        member.failing_code = failing_code
        return member

    # to start serving: faster than what serves today, by dispatch's default bar: a kernel validation admits is
    # one it serves
    ENTRY = "entry", DEFAULT_MIN_SPEEDUP, FindingCode.NOT_FASTER
    # to keep serving: a small slowdown over the Baseline allowed
    RETAIN = "retain", 0.99, FindingCode.REGRESSED

    def passes(self, ratio: float) -> bool:
        """Whether a ratio's lower bound clears this threshold."""
        return ratio > self.bar
