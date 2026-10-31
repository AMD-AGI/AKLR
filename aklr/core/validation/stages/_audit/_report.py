# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal

from aklr.core.validation.stages.stage import StageReport

__all__ = ["AuditReport"]


class AuditReport(StageReport):
    """The audit's findings."""

    stage_name: Literal["audit"] = "audit"
