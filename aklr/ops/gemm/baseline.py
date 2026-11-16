###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the gemm op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations


def gemm_a16w16_baseline(a, b, bias=None, out_dtype=None, use_shuffled_weight=False, **_):
    from aiter.tuned_gemm import gemm_a16w16

    b.is_shuffled = use_shuffled_weight
    return lambda: gemm_a16w16(A=a, B=b, bias=bias, otype=out_dtype)


def gemm_a8w8_baseline(
    a,
    b,
    a_scale,
    b_scale,
    bias=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8

    if use_shuffled_weight:
        return None
    return lambda: gemm_a8w8(XQ=a, WQ=b, x_scale=a_scale, w_scale=b_scale, bias=bias, dtype=out_dtype)


def _gemm_a8w8_blockscale_baseline(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale

    if scale_major_mode != "K" or use_shuffled_weight:
        return None
    return lambda: gemm_a8w8_blockscale(XQ=a, WQ=b, x_scale=a_scale, w_scale=b_scale, dtype=out_dtype)


def _triton_gemm_a8w8_blockscale_baseline(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    if scale_major_mode != "K" or use_shuffled_weight:
        return None
    return lambda: gemm_a8w8_blockscale(x=a, w=b, x_scale=a_scale, w_scale=b_scale, dtype=out_dtype, y=out)


def _gemm_a8w8_blockscale_bpreshuffle_baseline(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

    if not use_shuffled_weight:
        return None
    if scale_major_mode == "MN":
        return lambda: gemm_a8w8_blockscale_bpreshuffle(
            XQ=a, WQ=b, x_scale=a_scale.mT, w_scale=b_scale, dtype=out_dtype, out=out
        )
    # a K-major a_scale is relaid out in the timed call: aiter takes the MN-major one
    return lambda: gemm_a8w8_blockscale_bpreshuffle(
        XQ=a,
        WQ=b,
        x_scale=a_scale.mT.contiguous().mT,
        w_scale=b_scale,
        dtype=out_dtype,
        out=out,
    )


def gemm_fp8_nt_blockscaled_baseline(**call):
    """aiter's CK blockscale GEMM, else its Triton one, else its CK preshuffled one: the first that takes the call."""
    for baseline in (
        _gemm_a8w8_blockscale_baseline,
        _triton_gemm_a8w8_blockscale_baseline,
        _gemm_a8w8_blockscale_bpreshuffle_baseline,
    ):
        timed = baseline(**call)
        if timed is not None:
            return timed
    return None


def gemm_a4w4_baseline(
    a,
    b,
    a_descale,
    b_descale,
    bias=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4

    if not use_shuffled_weight:
        return None
    return lambda: gemm_a4w4(A=a, B=b, A_scale=a_descale, B_scale=b_descale, bias=bias, dtype=out_dtype)


def batched_gemm_bf16_baseline(A, B, out_dtype=None, use_shuffled_weight=False, **_):
    from aiter.ops.batched_gemm_op_bf16 import batched_gemm_bf16_CK

    if use_shuffled_weight:
        return None
    return lambda: batched_gemm_bf16_CK(XQ=A, WQ=B, dtype=out_dtype)
