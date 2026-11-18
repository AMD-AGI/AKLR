# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["per_group_quant_fp8_reference", "per_group_quant_mxfp8_reference"]


def per_group_quant_fp8_reference(x) -> tuple[Tensor, Tensor]:
    """The FP8 values and group-major scales in float64.

    The scale multiplication, reciprocal and scaled activation each round to float32, as
    AITER's group quantization does; one division instead changes quantization ties.
    """
    import torch

    x = x.double()
    rows, columns = x.shape
    grouped = x.reshape(rows, columns // 128, 128)
    maximum = grouped.abs().amax(-1).clamp_min(float(torch.tensor(1e-10, dtype=torch.float32)))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    scales = (maximum * divisor).to(torch.float32).double()
    reciprocal = (1.0 / scales).to(torch.float32).double()
    values = (grouped * reciprocal.unsqueeze(-1)).to(torch.float32).double().clamp(-448.0, 448.0)
    # FP8 rounding is the operation itself, not an implementation's error
    quantized = values.to(torch.float8_e4m3fn).double().reshape(rows, columns)
    return quantized, scales.t().contiguous()


def per_group_quant_mxfp8_reference(x) -> tuple[Tensor, Tensor]:
    """The scale multiply rounds to float32, then its exponent rounds up; the division by
    that power of two and the FP8 rounding are the quantization itself."""
    import torch

    x = x.double()
    rows, columns = x.shape
    grouped = x.reshape(rows, columns // 32, 32)
    maximum = grouped.abs().amax(-1).clamp_min(float(torch.tensor(1e-10, dtype=torch.float32)))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    raw = (maximum * divisor).to(torch.float32).contiguous().view(torch.int32)
    exponent = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).contiguous().view(torch.float32).double()
    values = (grouped / scale.unsqueeze(-1)).to(torch.float32).clamp(-448.0, 448.0)
    return values.to(torch.float8_e4m3fn).double().reshape(rows, columns), exponent.to(torch.uint8)
