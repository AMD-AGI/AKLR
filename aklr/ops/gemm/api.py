# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route, must_default, unused
from aklr.ops.gemm.contract import (
    BMM,
    GEMM,
    GEMM_A4W4_MXFP4,
    GEMM_A8W8,
    GEMM_A8W8_BLOCKWISE_SCALED,
)

__all__ = [
    "bmm",
    "gemm",
    "gemm_a4w4_mxfp4",
    "gemm_a8w8",
    "gemm_a8w8_blockwise_scaled",
]


@api(GEMM)
def gemm(a, b, trans_b=True, bias=None, out_dtype=None, out=None): ...


@api(GEMM_A8W8)
def gemm_a8w8(a, b, a_scale, b_scale, trans_b=True, bias=None, out_dtype=None, out=None): ...


@api(GEMM_A8W8_BLOCKWISE_SCALED)
def gemm_a8w8_blockwise_scaled(
    a,
    b,
    a_scale,
    b_scale,
    block_size=(128, 128),
    scale_major_mode="K",
    trans_b=True,
    bias=None,
    out_dtype=None,
    out=None,
    *,
    operand_format="none:none",
): ...


@api(GEMM_A4W4_MXFP4)
def gemm_a4w4_mxfp4(
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
): ...


@api(BMM)
def bmm(a, b, trans_b=True, bias=None, out_dtype=None, out=None): ...


@aklr_route("aiter.tuned_gemm.gemm_a16w16")
def _(A, B, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
    must_default(scale_a, scale_b, scale_c)  # a scaled gemm is not this op
    return gemm(a=A, b=B, bias=bias, out_dtype=otype)


@aklr_route("aiter.ops.gemm_op_a8w8.gemm_a8w8")
def _(XQ, WQ, x_scale, w_scale, bias=None, dtype=None, splitK=None):
    unused(splitK)  # scheduling; the registry decides
    return gemm_a8w8(a=XQ, b=WQ, a_scale=x_scale, b_scale=w_scale, bias=bias, out_dtype=dtype)


# FIXME(main-kernels): the bench baseline is the first route that binds; main's plain
# blockscale baseline is CK, so CK comes before Triton.
@aklr_route("aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale")
def _(XQ, WQ, x_scale, w_scale, dtype=None, isBpreshuffled=False):
    # This flag selects a different gfx950 ASM packing, not the 16x16 one below.
    must_default(isBpreshuffled)
    return gemm_a8w8_blockwise_scaled(a=XQ, b=WQ, a_scale=x_scale, b_scale=w_scale, out_dtype=dtype)


@aklr_route("aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale.gemm_a8w8_blockscale")
def _(
    x,
    w,
    x_scale,
    w_scale,
    dtype=None,
    y=None,
    config=None,
    skip_reduce=False,
    kernel_type="bandwidth_bound",
    backend=None,
):
    must_default(config, skip_reduce, kernel_type, backend)  # explicit tuning
    return gemm_a8w8_blockwise_scaled(a=x, b=w, a_scale=x_scale, b_scale=w_scale, out_dtype=dtype, out=y)


@aklr_route("aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle")
def _(XQ, WQ, x_scale, w_scale, dtype=None, out=None):
    return gemm_a8w8_blockwise_scaled(
        a=XQ,
        b=WQ,
        a_scale=x_scale,
        b_scale=w_scale,
        operand_format="none:preshuffled",
        out_dtype=dtype,
        out=out,
    )


@aklr_route("aiter.ops.gemm_op_a4w4.gemm_a4w4")
def _(
    A,
    B,
    A_scale,
    B_scale,
    bias=None,
    dtype=None,
    alpha=1.0,
    beta=0.0,
    bpreshuffle=True,
    apreshuffle=False,
    global_A_scale=None,
    global_B_scale=None,
):
    # aiter's one layout: b and both scales shuffled; anything else is not described.
    must_default(bpreshuffle, apreshuffle)
    must_default(global_A_scale, global_B_scale)  # set means nvfp4, another op
    return gemm_a4w4_mxfp4(
        a=A,
        b=B,
        a_scale=A_scale,
        b_scale=B_scale,
        operand_format="none:preshuffled",
        alpha=alpha,
        beta=beta,
        bias=bias,
        out_dtype=dtype,
    )


@aklr_route("aiter.ops.batched_gemm_op_bf16.batched_gemm_bf16_CK")
def _(XQ, WQ, bias=None, dtype=None, splitK=None):
    unused(splitK)  # scheduling; the registry decides
    return bmm(a=XQ, b=WQ, bias=bias, out_dtype=dtype)
