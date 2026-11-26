# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["DeepseekV4WoAMxfp8Reference"]


class DeepseekV4WoAMxfp8Reference(OpReference):
    def reference(self, *, o, wo_a, **_):
        """The grouped product in float32, rounded to bf16 as wo_a's output is; then the per-32 fp8
        e4m3 quantization wo_b's operand takes: the scale multiply rounds to float32, its exponent
        rounds upward and is clamped to [1, 254], the codes are the values over that power of two,
        clamped to +-448 and rounded to e4m3."""
        y = torch.einsum("tgd,grd->tgr", o.float(), wo_a.float()).to(torch.bfloat16).float()
        rows = y.shape[0]
        grouped = y.reshape(rows, -1, 32)
        maximum = grouped.abs().amax(-1).clamp_min(1e-10)
        raw = (maximum * (1.0 / 448.0)).contiguous().view(torch.int32)
        exponent = ((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
        exponent = exponent.clamp(1, 254)
        scale = (exponent << 23).contiguous().view(torch.float32)
        values = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
        quantized = values.to(torch.float8_e4m3fn).reshape(rows, -1)
        return {"quantized": quantized, "scales": exponent.to(torch.uint8)}
