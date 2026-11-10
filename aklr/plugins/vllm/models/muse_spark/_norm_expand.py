###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright contributors to the vLLM project
#
# Adapted from vLLM (https://github.com/vllm-project/vllm)
# Source: vllm/models/muse_spark/amd/ops/fused_norm_expand.py at 4fab5a074
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/plugins/vllm/models/muse_spark: the kernel as is, its launcher returning
# the expanded rows
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

"""vLLM's pre-expert RMSNorm of Muse Spark, written once per route, for the baseline to run
without vLLM."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["rmsnorm_expand"]


@triton.jit
def _rmsnorm_expand_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    eps,
    gain_center,
    REPEATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    x = tl.load(x_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center
        if INPUT_SCALE:
            x = x * w
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    if HAS_WEIGHT and not INPUT_SCALE:
        y = y * w
    y = y.to(out_ptr.dtype.element_ty)
    for r in tl.static_range(REPEATS):
        tl.store(out_ptr + (t * REPEATS + r) * D + d, y, mask=dm)


def rmsnorm_expand(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    gain_center: float,
    input_scale: bool,
    repeats: int,
) -> torch.Tensor:
    """RMSNorm of ``x`` written ``repeats`` times per row: ``[T * repeats, D]``."""
    assert x.dim() == 2 and x.is_contiguous()
    T, D = x.shape
    out = torch.empty((T, repeats, D), dtype=x.dtype, device=x.device)
    if T > 0:
        block_d = triton.next_power_of_2(D)
        _rmsnorm_expand_kernel[(T,)](
            x,
            weight if weight is not None else x,
            out,
            D=D,
            eps=eps,
            gain_center=gain_center,
            REPEATS=repeats,
            BLOCK_D=block_d,
            HAS_WEIGHT=weight is not None,
            INPUT_SCALE=input_scale and weight is not None,
            num_warps=8 if block_d >= 4096 else 4,
        )
    return out.view(-1, D)
