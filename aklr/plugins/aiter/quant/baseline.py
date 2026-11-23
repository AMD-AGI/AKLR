# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


def per_group_quant_fp8_baseline(x, *, quantized=None, scales=None):
    import torch
    from aiter.ops.quant import per_group_quant_hip

    quantized, raw_scales = per_group_quant_hip(
        x, group_size=128, quant_dtype=torch.float8_e4m3fn, transpose_scale=True
    )
    # AITER reports [rows, groups] but stores group-major bytes.
    return quantized, raw_scales.view(x.shape[1] // 128, x.shape[0])


def per_group_quant_mxfp8_baseline(x, *, quantized=None, scales=None):
    import torch
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    quantized, raw_scales = per_1x32_mx_quant_hip(
        x, quant_dtype=torch.float8_e4m3fn, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    return quantized, raw_scales.view(torch.uint8)
