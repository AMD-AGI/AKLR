# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fp8 GEMMs with a fp32 scale per 128 x 128 of the weight and per 1 x 128 of the
activations, served by AKLR's ``gemm_a8w8_blockwise_scaled``. aiter takes each layout of the weight
as its own op, AKLR as an ``operand_format`` of one contract; each op keeps its own ``run``."""

from __future__ import annotations

import torch

from aklr.ops.gemm.api import gemm_a8w8_blockwise_scaled
from aklr.plugins.aiter.provider._base import offer, selector, served


def _fp32_scales(x_scale, w_scale) -> bool:
    """aiter also takes e8m0 scales per 32 (group32); AKLR's contract has fp32 per 128 only."""
    return x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32


@selector("gemm_a8w8_blockscale")
def _select_gemm_a8w8_blockscale(args, kwargs):
    xq, wq, x_scale, w_scale, dtype, is_b_preshuffled, split_k = args
    # a shuffled weight is gemm_a8w8_blockscale_bpreshuffle's contract (aiter has no fp32 kernel
    # for it behind this op on gfx950); split_k has no counterpart in AKLR's
    if is_b_preshuffled or split_k is not None or not _fp32_scales(x_scale, w_scale):
        return None
    return offer(
        gemm_a8w8_blockwise_scaled.serving(xq, wq, x_scale, w_scale, out_dtype=dtype),
        _run_gemm_a8w8_blockscale,
    )


def _run_gemm_a8w8_blockscale(xq, wq, x_scale, w_scale, dtype, is_b_preshuffled, split_k):
    return served(gemm_a8w8_blockwise_scaled(xq, wq, x_scale, w_scale, out_dtype=dtype))


@selector("gemm_a8w8_blockscale_bpreshuffle")
def _select_gemm_a8w8_blockscale_bpreshuffle(args, kwargs):
    xq, wq, x_scale, w_scale, dtype, out = args
    # aiter reads x_scale's bytes column-major; AKLR's is_row_major marker would read them row-major
    if not _fp32_scales(x_scale, w_scale) or getattr(x_scale, "is_row_major", False):
        return None
    # x_scale holds aiter's column-major bytes, AKLR's default for a shuffled weight
    return offer(
        gemm_a8w8_blockwise_scaled.serving(
            xq, wq, x_scale, w_scale, out_dtype=dtype, out=out, operand_format="none:preshuffled"
        ),
        _run_gemm_a8w8_blockscale_bpreshuffle,
    )


def _run_gemm_a8w8_blockscale_bpreshuffle(xq, wq, x_scale, w_scale, dtype, out):
    return served(
        gemm_a8w8_blockwise_scaled(
            xq, wq, x_scale, w_scale, out_dtype=dtype, out=out, operand_format="none:preshuffled"
        )
    )
