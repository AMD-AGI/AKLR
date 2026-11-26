# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["Mxfp8E4m3QuantizeReference"]

_FLT_MIN = torch.finfo(torch.float32).tiny


class Mxfp8E4m3QuantizeReference(OpReference):
    def reference(self, *, x, **_):
        """The scale byte is the exact ceil(log2(amax / 448)) + 127, from the float32 bits of the amax: a BF16
        amax carries 7 mantissa bits, so its quotient by 448 is a power of two only when the amax's mantissa
        is 1.75, and lies at least 2**-7 / 1.75 above one otherwise; float32 division and log2 then round to
        the same ceil. The rescale by the exact power of two and the final FP8 rounding are the operation."""
        x = x.float()
        rows, columns = x.shape
        grouped = x.reshape(rows, columns // 32, 32)
        amax = grouped.abs().amax(-1).clamp_min(_FLT_MIN).contiguous()
        bits = amax.view(torch.int32)
        # 0x600000 is the mantissa field of 1.75 = 448 / 2**8
        byte = (bits >> 23) - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
        byte = byte.clamp(0, 254)
        rescale = ((254 - byte) << 23).contiguous().view(torch.float32)
        quantized = (grouped * rescale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(rows, columns)
        return {"quantized": quantized, "scales": byte.to(torch.uint8)}
