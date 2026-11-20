# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's block-scaled fp8 GEMM on a shuffled weight, ``aiter.gemm_a8w8_blockscale_bpreshuffle``,
served by AKLR's ``gemm_a8w8_blockwise_scaled`` op."""

from __future__ import annotations

from aklr.ops.gemm.api import gemm_a8w8_blockwise_scaled

__all__ = ["gemm_a8w8_blockscale_bpreshuffle"]


def gemm_a8w8_blockscale_bpreshuffle(XQ, WQ, x_scale, w_scale, *, dtype, out):
    """AKLR's block-scaled ``XQ @ WQ.T`` into ``out``, or why aiter computes it: ``WQ`` is
    shuffled 16 x 16 and ``x_scale`` holds its bytes column by column, as aiter reads them."""
    return gemm_a8w8_blockwise_scaled(
        XQ, WQ, x_scale, w_scale, out_dtype=dtype, out=out, operand_format="none:preshuffled"
    )
