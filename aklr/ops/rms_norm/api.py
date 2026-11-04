###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.rms_norm.schema import ADD_RMS_NORM, RMS_NORM

__all__ = ["add_rms_norm", "rms_norm"]


@aklr_api(ADD_RMS_NORM)
def add_rms_norm(x, residual, weight, epsilon, normalized, residual_out): ...


@aklr_api(RMS_NORM)
def rms_norm(x, weight, epsilon, normalized=None): ...
