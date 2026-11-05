# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent arithmetic specification of the group-128 quantizer."""


def per_group_quant_fp8_reference(point, row_major_args, to_float64):
    """Return the quantized FP8 values and group-major scales in float64.

    The scale multiplication, reciprocal and scaled activation each round to
    float32, as required by the AITER group quantization operation. Combining
    the scale and reciprocal into one division changes quantization ties.
    """
    import torch

    x = to_float64(row_major_args["x"], point.dtypes["x"])
    rows, columns = x.shape
    grouped = x.reshape(rows, columns // 128, 128)
    maximum = grouped.abs().amax(-1).clamp_min(float(torch.tensor(1e-10, dtype=torch.float32)))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    scales = (maximum * divisor).to(torch.float32).double()
    reciprocal = (1.0 / scales).to(torch.float32).double()
    values = (grouped * reciprocal.unsqueeze(-1)).to(torch.float32).double().clamp(-448.0, 448.0)
    # FP8 rounding is the operation itself, rather than an implementation's
    # numerical error against an unquantized real-valued output.
    quantized = values.to(torch.float8_e4m3fn).double().reshape(rows, columns)
    return quantized, scales.t().contiguous()


def per_group_quant_mxfp8_reference(point, row_major_args, to_float64):
    """The scale multiply rounds to float32, then its exponent rounds upward.

    The division by that exact power of two and final FP8 rounding are
    intrinsic to the quantization operation, including the 1e-10 amax floor.
    """
    import torch

    x = to_float64(row_major_args["x"], point.dtypes["x"])
    rows, columns = x.shape
    grouped = x.reshape(rows, columns // 32, 32)
    maximum = grouped.abs().amax(-1).clamp_min(float(torch.tensor(1e-10, dtype=torch.float32)))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    raw = (maximum * divisor).to(torch.float32).contiguous().view(torch.int32)
    exponent = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).contiguous().view(torch.float32).double()
    values = (grouped / scale.unsqueeze(-1)).to(torch.float32).clamp(-448.0, 448.0)
    return values.to(torch.float8_e4m3fn).double().reshape(rows, columns), exponent.double()
