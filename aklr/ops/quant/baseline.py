###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the quant op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import torch


def per_group_quant_hip_baseline(x, **_):
    from aiter.ops.quant import per_group_quant_hip

    return lambda: per_group_quant_hip(
        x, group_size=128, quant_dtype=torch.float8_e4m3fn, transpose_scale=True
    )


def per_1x32_mx_quant_hip_baseline(x, **_):
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    return lambda: per_1x32_mx_quant_hip(
        x, quant_dtype=torch.float8_e4m3fn, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
