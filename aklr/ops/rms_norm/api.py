# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.rms_norm.contract import ADD_RMS_NORM, RMS_NORM

__all__ = ["add_rms_norm", "rms_norm"]


@api(ADD_RMS_NORM)
def add_rms_norm(x, residual, weight, epsilon, normalized, residual_out): ...


@api(RMS_NORM)
def rms_norm(x, weight, epsilon): ...
