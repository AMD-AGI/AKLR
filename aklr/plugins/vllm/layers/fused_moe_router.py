# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE routing on ROCm (``model_executor/layers/fused_moe/router``): the gate GEMM with an fp32
output that gfx950 runs for a few tokens (Kimi-K3's router)."""

from __future__ import annotations

import os

import torch

from aklr.core.contract.fastkey._fast_key_fn import MIN_PTR_ALIGN_BYTES
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import _original, vllm_route

__all__: list[str] = []

_VERSIONS = ("0.34.0", "0.32.1rc1.dev276")
_CALLER = "vllm.model_executor.layers.fused_moe.router.gate_linear.fp32_router_gemm_dispatch_impl"
_ROUTER = "vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm"


@vllm_route(_CALLER, f"{_ROUTER}.rocm_fp32_router_gemm", versions=_VERSIONS)
def _rocm_fp32_router_gemm(hidden_states, router_weight):
    """The gate's ``hidden_states @ router_weight.T`` in fp32: vLLM's own kernel where vLLM takes the call (at
    most 9 tokens for Kimi-K3's 896 bf16 experts), else AKLR's, which ``_can_use_rocm_fp32_router_gemm`` let in."""
    if _native_can_use(hidden_states, router_weight):
        # AKLR's gemm is qualified against aiter's GEMM, not against this kernel of vLLM's
        return FallbackReason.NOT_QUALIFIED
    out = gemm(hidden_states, router_weight, out_dtype=torch.float32)
    # a call AKLR declines after all (a remembered answer of another call): vLLM's kernel would reject it
    return _unrouted(hidden_states, router_weight) if isinstance(out, FallbackReason) else out


@vllm_route(_CALLER, f"{_ROUTER}.can_use_rocm_fp32_router_gemm", versions=_VERSIONS)
def _can_use_rocm_fp32_router_gemm(hidden_states, router_weight):
    """Whether vLLM takes its fp32 router GEMM: vLLM's own answer where it takes the call itself, else yes where
    AKLR serves it. For Kimi-K3's router vLLM's kernel stops at 9 tokens (hipBLASLt from 10, its decode's 16
    tokens included); AKLR's gemm serves the counts its workloads measured faster."""
    if _native_can_use(hidden_states, router_weight):
        return FallbackReason.NOT_QUALIFIED  # vLLM's own answer: yes
    key = (
        tuple(hidden_states.shape), hidden_states.stride(), hidden_states.dtype, hidden_states.device,
        hidden_states.data_ptr() % MIN_PTR_ALIGN_BYTES == 0,
        tuple(router_weight.shape), router_weight.stride(), router_weight.dtype,
        router_weight.data_ptr() % MIN_PTR_ALIGN_BYTES == 0,
    )  # fmt: skip
    reason = _REASONS.get(key, _UNASKED)
    if reason is _UNASKED:
        # one dispatch lookup per call shape: vLLM asks on every MoE layer, every step
        reason = gemm.fallback_reason(hidden_states, router_weight, out_dtype=torch.float32)
        # a capture's not-warmed answer changes once the call runs outside one
        if _REMEMBER and reason is not FallbackReason.NOT_WARMED:
            _REASONS[key] = reason
    return True if reason is None else reason


def _native_can_use(hidden_states, router_weight) -> bool:
    """vLLM's own answer: whether its fp32 router kernel takes the call."""
    return _original(_can_use_rocm_fp32_router_gemm)(hidden_states, router_weight)


def _unrouted(hidden_states, router_weight):
    """What vLLM's fp32_router_gemm_dispatch_impl runs when its fp32 router kernel does not take the call."""
    if router_weight.dtype == torch.bfloat16:
        return torch.mm(hidden_states, router_weight.T, out_dtype=torch.float32)
    return torch.nn.functional.linear(hidden_states.float(), router_weight)


# gemm's answer per call shape (None: served); not kept while exporting, which counts every call
_REASONS: dict[tuple, FallbackReason | None] = {}
_UNASKED = object()
_REMEMBER = os.environ.get("AKLR_EXPORT") != "1"
