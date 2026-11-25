# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.hook import aiter_asks_aklr, vllm_route

__all__: list[str] = []

# aiter's GEMM is AKLR's only through AKLR's aiter provider: without it vLLM keeps its own
_IS_PROVIDER_ON = aiter_asks_aklr()


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "vllm._custom_ops.wvSplitKrc",
    versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
)
def _wv_split_krc(a, b, cu_count, bias=None):
    """vLLM's skinny GEMM on gfx950 for 10 to 128 tokens as aiter's GEMM, which AKLR's aiter provider answers."""
    if not _IS_PROVIDER_ON:
        return FallbackReason.DISABLED
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(a, b, bias)
