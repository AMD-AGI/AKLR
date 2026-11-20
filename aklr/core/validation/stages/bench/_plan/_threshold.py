###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


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

    ENTRY = "entry", DEFAULT_MIN_SPEEDUP, FindingCode.NOT_FASTER
    RETAIN = "retain", 0.99, FindingCode.REGRESSED

    def passes(self, ratio: float) -> bool:
        return ratio > self.bar
