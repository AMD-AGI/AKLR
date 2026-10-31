# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Base of every contract data class: immutable, unknown fields rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")
