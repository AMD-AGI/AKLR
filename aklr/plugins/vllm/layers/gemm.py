# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's unquantized linear layers on ROCm, where they reach aiter's tuned GEMM or vLLM's
skinny GEMM."""

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm, gemm_a8w8
from aklr.plugins import aiter as aklr_aiter
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "aiter.tuned_gemm.tgemm.mm",
    versions=("0.33.0rc1.dev910",),
)
def _tgemm_mm(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
    return aklr_aiter.mm(
        inp, weights, bias=bias, otype=otype, scale_a=scale_a, scale_b=scale_b, scale_c=scale_c, out=out
    )


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "vllm._custom_ops.wvSplitKrc",
    versions=("0.33.0rc1.dev910",),
)
def _wv_split_krc(a, b, cu_count, bias=None):
    """vLLM's skinny GEMM on gfx950 for 10 to 128 tokens, ahead of aiter where it fits."""
    return gemm(a, b, bias=bias)


@vllm_route(
    "vllm._aiter_ops._rocm_aiter_preshuffled_per_token_w8a8_gemm_impl",
    "aiter.gemm_a8w8_bpreshuffle",
    versions=("0.33.0rc1.dev910",),
)
def _gemm_a8w8_bpreshuffle(*args, **kwargs):
    """vLLM's per-token fp8 linears on preshuffled weights (Kimi-K3's KDA in-projection above one request).
    aiter exports the function behind a guard taking ``(*args, **kwargs)``; the parameters are those of
    ``aiter.ops.gemm_op_a8w8.gemm_a8w8_bpreshuffle``."""
    return _preshuffled_per_token_gemm(*args, **kwargs)


def _preshuffled_per_token_gemm(XQ, WQ, x_scale, w_scale, bias=None, dtype=torch.bfloat16, check=False):
    # the caller is the preshuffled path, so WQ is shuffled whether or not it still carries the mark: vLLM
    # re-wraps it as a Parameter, which drops aiter's is_shuffled
    if check:
        return FallbackReason.OUT_OF_CONTRACT
    b = WQ.view(WQ.shape)
    b.is_shuffled = True
    return gemm_a8w8(XQ, b, x_scale, w_scale, bias=bias, out_dtype=dtype)
