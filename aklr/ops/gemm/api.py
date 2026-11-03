# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
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
