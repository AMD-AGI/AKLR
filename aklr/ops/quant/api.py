# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.runtime.dispatch import api
from aklr.ops.quant.contract import PER_GROUP_QUANT_FP8, PER_GROUP_QUANT_MXFP8


@api(PER_GROUP_QUANT_FP8)
def per_group_quant_fp8(x): ...


@api(PER_GROUP_QUANT_MXFP8)
def per_group_quant_mxfp8(x): ...
