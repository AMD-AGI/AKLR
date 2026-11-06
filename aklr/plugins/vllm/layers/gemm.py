# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's unquantized linear layers on ROCm, where they reach aiter's tuned GEMM."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "aiter.tuned_gemm.tgemm.mm",
    versions=("0.33.0rc1.dev910",),
)
def _tgemm_mm(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
    if scale_a is not None or scale_b is not None or scale_c is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return gemm(inp, weights, bias=bias, out_dtype=otype, out=out)
