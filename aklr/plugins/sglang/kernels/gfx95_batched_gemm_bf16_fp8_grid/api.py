# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.kernels.gfx95_batched_gemm_bf16_fp8_grid.contract import (
    DEEPSEEK_V4_WO_A_SPLIT_K,
    DEEPSEEK_V4_WO_A_SPLIT_K_MXFP8,
)

__all__ = ["deepseek_v4_wo_a_split_k", "deepseek_v4_wo_a_split_k_mxfp8"]


@api(DEEPSEEK_V4_WO_A_SPLIT_K)
def deepseek_v4_wo_a_split_k(x, w, fp8_grid=False, out_dtype=None, out=None): ...


@api(DEEPSEEK_V4_WO_A_SPLIT_K_MXFP8)
def deepseek_v4_wo_a_split_k_mxfp8(x, w, quantized=None, scales=None): ...


@sglang_route(
    "sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid._batched_gemm_split_k",
    versions=("0.5.23",),
)
def _batched_gemm_split_k(x, w, out, fp8_grid, scale=None):
    """The split-K regime of sglang's gfx950 wo_a GEMM (DeepSeek-V4.1's decode and verify rows,
    at most 64), which batched_gemm_bf16_fp8_grid looks up at call time, as one op call into
    sglang's out: deepseek_v4_wo_a_split_k for a bf16 out (on the fp8 grid when fp8_grid), and
    deepseek_v4_wo_a_split_k_mxfp8 when sglang passes scale for the fp8 codes' e8m0 bytes (fp8_grid
    is then unused, as in sglang). The function reads no sglang configuration: shapes, dtypes and
    layouts the ops do not cover fall back through them."""
    if scale is None:
        result = deepseek_v4_wo_a_split_k(x=x, w=w, fp8_grid=bool(fp8_grid), out=out)
    else:
        result = deepseek_v4_wo_a_split_k_mxfp8(x=x, w=w, quantized=out, scales=scale)
    return result if isinstance(result, FallbackReason) else None
