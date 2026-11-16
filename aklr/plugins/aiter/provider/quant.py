# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's group quantization, ``per_group_quant``, served by AKLR's ``per_group_quant_fp8``
(groups of 128, fp32 scales in aiter's transposed bytes) and ``per_group_quant_mxfp8`` (groups of
32, row-major e8m0 scales)."""

from __future__ import annotations

import torch

from aklr.ops.quant.api import per_group_quant_fp8, per_group_quant_mxfp8
from aklr.plugins.aiter.provider._base import offer, selector, served

_E8M0 = getattr(torch, "float8_e8m0fnu", None)


@selector("per_group_quant")
def _select_per_group_quant(args, kwargs):
    x, scale, quant_dtype, group_size, transpose_scale, num_rows, _, scale_type, scale_layout_m32k4 = args
    # num_rows_factor only acts with num_rows
    if (
        scale is not None
        or num_rows is not None
        or scale_layout_m32k4
        or quant_dtype != torch.float8_e4m3fn
        or x.dim() != 2
    ):
        return None
    if group_size == 128 and transpose_scale and scale_type == torch.float32:
        return offer(per_group_quant_fp8.serving(x), _run_per_group_quant_fp8)
    if group_size == 32 and not transpose_scale and scale_type == _E8M0:
        return offer(per_group_quant_mxfp8.serving(x), _run_per_group_quant_mxfp8)
    return None


def _run_per_group_quant_fp8(
    x,
    scale,
    quant_dtype,
    group_size,
    transpose_scale,
    num_rows,
    num_rows_factor,
    scale_type,
    scale_layout_m32k4,
):
    quantized, scales = served(per_group_quant_fp8(x))
    # aiter shapes its transposed scales (rows, groups) over group-major bytes: AKLR's (groups, rows)
    return quantized, scales.view(x.shape[0], x.shape[1] // 128)


def _run_per_group_quant_mxfp8(
    x,
    scale,
    quant_dtype,
    group_size,
    transpose_scale,
    num_rows,
    num_rows_factor,
    scale_type,
    scale_layout_m32k4,
):
    quantized, scales = served(per_group_quant_mxfp8(x))
    return quantized, scales.view(_E8M0)
