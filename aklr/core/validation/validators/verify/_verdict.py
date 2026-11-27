# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

from typing import Literal

from aklr.core.common.frozen_model import frozen_dataclass

__all__ = ["Verdict", "VerdictStatus"]

VerdictStatus = Literal["proved", "rejected", "unchecked"]


@frozen_dataclass
class Verdict:
    rule: str
    status: VerdictStatus
    detail: str = ""

    @property
    def reason(self) -> str:
        return self.detail.partition(":")[0]
