# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = [
    "bmm_baseline",
    "gemm_a4w4_mxfp4_baseline",
    "gemm_a8w8_baseline",
    "gemm_a8w8_blockwise_scaled_baseline",
    "gemm_baseline",
]


def gemm_baseline(a, b, trans_b=True, bias=None, out_dtype=None, out=None):
    """aiter's tuned 16-bit gemm, b as (n, k) only."""
    from aiter.tuned_gemm import gemm_a16w16

    if not trans_b:
        raise NotImplementedError("aiter's gemm_a16w16 takes b as (n, k) only")
    return gemm_a16w16(a, b, bias=bias, otype=out_dtype)


def gemm_a8w8_baseline(a, b, a_scale, b_scale, trans_b=True, bias=None, out_dtype=None, out=None):
    """aiter's CK 8-bit gemm, b as (n, k) only."""
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8

    if not trans_b:
        raise NotImplementedError("aiter's gemm_a8w8 takes b as (n, k) only")
    return gemm_a8w8(a, b, a_scale, b_scale, bias=bias, dtype=out_dtype)


def gemm_a8w8_blockwise_scaled_baseline(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="K",
    trans_b=True,
    bias=None,
    out_dtype=None,
    out=None,
    *,
    operand_format="none:none",
):
    """aiter's CK blockscale gemm, or with a shuffled b its bpreshuffle gemm. That one
    reads a_scale's bytes column by column whatever its strides, so an a_scale marked
    ``is_row_major`` is relaid in the timed call, as main's baseline does."""
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale,
        gemm_a8w8_blockscale_bpreshuffle,
    )

    if scale_major_mode != "K" or not trans_b or bias is not None:
        raise NotImplementedError("aiter's blockscale gemm: K-major a_scale, b as (n, k), no bias")
    if operand_format == "none:none":
        return gemm_a8w8_blockscale(a, b, a_scale, b_scale, dtype=out_dtype)
    if getattr(a_scale, "is_row_major", False):
        a_scale = a_scale.T.contiguous().T
    return gemm_a8w8_blockscale_bpreshuffle(a, b, a_scale, b_scale, dtype=out_dtype, out=out)


def gemm_a4w4_mxfp4_baseline(
    a,
    b,
    a_scale,
    b_scale,
    alpha=1.0,
    beta=0.0,
    trans_b=True,
    bias=None,
    out_dtype=None,
    out=None,
    *,
    operand_format="none:preshuffled",
):
    """aiter's mxfp4 gemm, on its one layout: b and both scales shuffled."""
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4

    if not trans_b or operand_format != "none:preshuffled":
        raise NotImplementedError("aiter's gemm_a4w4 takes a shuffled (n, k) b only")
    return gemm_a4w4(a, b, a_scale, b_scale, bias=bias, dtype=out_dtype, alpha=alpha, beta=beta)


def bmm_baseline(a, b, trans_b=True, bias=None, out_dtype=None, out=None):
    """aiter's CK 16-bit batched gemm when b is (batch, n, k); otherwise torch.bmm, which sglang runs for an MLA
    layer's absorbed w_kc and w_vc products and which aiter has no batched gemm for."""
    if not trans_b:
        import torch

        product = torch.bmm(a, b)
        if bias is not None:
            product = product + bias[:, None, :]
        return product.to(out_dtype or a.dtype)
    from aiter.ops.batched_gemm_op_bf16 import batched_gemm_bf16_CK

    return batched_gemm_bf16_CK(a, b, bias=bias, dtype=out_dtype)
