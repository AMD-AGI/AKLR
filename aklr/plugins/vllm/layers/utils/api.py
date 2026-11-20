# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.plugins.aiter.provider.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "vllm._custom_ops.wvSplitKrc",
    versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
)
def _wv_split_krc(a, b, cu_count, bias=None):
    """vLLM's skinny GEMM on gfx950 for 10 to 128 tokens as aiter's GEMM, which AKLR's aiter provider answers."""
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(a, b, bias)


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "vllm._custom_ops.wvSplitK",
    versions=("0.33.0rc1.dev910", "0.34.0"),
)
def _wv_split_k(a, b, cu_count, bias=None):
    """vLLM's skinny GEMM for up to 5 rows (a prefill's 1-row LM head, DSpark's 5-row draft
    head), ahead of aiter; it takes the weight first, then the rows."""
    return gemm(b, a, bias=bias)
