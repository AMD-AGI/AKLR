###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright contributors to the vLLM project
#
# Adapted from vLLM (https://github.com/vllm-project/vllm)
# Source: vllm/models/kimi_k3/amd/ops/fp4_indexer.py at 14d1ec2336 (Kimi-K3.1-Pro's fork; not in a
# vLLM release)
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/plugins/vllm/models/kimi_k3/fp4_indexer: the query quantization kernel and its
# launcher as they are, without the key store and the decode scorer
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

"""vLLM's MXFP4 query quantization for Kimi-K3.1's indexer, for the baseline to run without the fork."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["quantize_queries"]


@triton.jit
def _quantize(values):
    values = tl.reshape(values, (4, 32))
    scale = tl.maximum(tl.max(tl.abs(values), 1) / 6.0, 1.0e-4)
    bits = scale.to(tl.int32, bitcast=True)
    exponent = tl.minimum(((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    x = values / scale[:, None]
    a = tl.abs(x)
    # E2M1 levels: 0, .5, 1, 1.5, 2, 3, 4, 6. Ties choose even codes.
    code = (a > 0.25).to(tl.uint8)
    code += (a >= 0.75).to(tl.uint8)
    code += (a > 1.25).to(tl.uint8)
    code += (a >= 1.75).to(tl.uint8)
    code += (a > 2.5).to(tl.uint8)
    code += (a >= 3.5).to(tl.uint8)
    code += (a > 5.0).to(tl.uint8)
    code |= ((x < 0) & (code != 0)).to(tl.uint8) << 3
    low, high = tl.split(tl.reshape(code, (64, 2)))
    return low | (high << 4), exponent.to(tl.uint8)


@triton.jit
def _quantize_query(Q, Payload, Scales, QS: tl.constexpr, QH: tl.constexpr, H: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    group = tl.arange(0, 4)
    scale_offsets = row * 256 + group * 64 + (head % 16) * 4 + head // 16
    if head < H:
        values = tl.load(Q + row * QS + head * QH + tl.arange(0, 128)).to(tl.float32)
        packed, scales = _quantize(values)
        tl.store(Payload + (row * H + head) * 64 + tl.arange(0, 64), packed)
        tl.store(Scales + scale_offsets, scales)
    else:
        tl.store(Scales + scale_offsets, tl.full((4,), 0, tl.uint8))


def quantize_queries(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [rows, heads, 128] without folding scales into head weights."""
    if q.ndim != 3 or q.shape[2] != 128 or q.stride(2) != 1:
        raise ValueError("FP4 index queries require contiguous 128-element heads")
    rows, heads, _ = q.shape
    if heads not in (16, 32, 64):
        raise ValueError("FP4 row-group queries support 16, 32 or 64 heads")
    payload = torch.empty((rows, heads, 64), device=q.device, dtype=torch.uint8)
    scales = torch.empty((rows, 1, 4, 16, 4), device=q.device, dtype=torch.uint8)
    if rows:
        _quantize_query[(rows, 64)](q, payload, scales, q.stride(0), q.stride(1), heads)
    return payload, scales
