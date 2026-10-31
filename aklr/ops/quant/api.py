# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.quant.schema import PER_GROUP_QUANT_FP8, PER_GROUP_QUANT_MXFP8

__all__ = ["per_group_quant_fp8", "per_group_quant_mxfp8"]


@aklr_api(PER_GROUP_QUANT_FP8)
def per_group_quant_fp8(x, quantized=None, scales=None): ...


@aklr_api(PER_GROUP_QUANT_MXFP8)
def per_group_quant_mxfp8(x, quantized=None, scales=None): ...
