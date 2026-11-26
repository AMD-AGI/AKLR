# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE routing on ROCm (``model_executor/layers/fused_moe/router``): the gate GEMM with an fp32
output that gfx950 runs for a few tokens (Kimi-K3's router)."""

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.hook import _original, vllm_route

__all__: list[str] = []

_VERSIONS = ("0.34.0", "0.32.1rc1.dev276")
_CALLER = "vllm.model_executor.layers.fused_moe.router.gate_linear.fp32_router_gemm_dispatch_impl"
_ROUTER = "vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm"


@vllm_route(_CALLER, f"{_ROUTER}.rocm_fp32_router_gemm", versions=_VERSIONS)
def _rocm_fp32_router_gemm(hidden_states, router_weight):
    """The gate's ``hidden_states @ router_weight.T`` in fp32: vLLM's own kernel where vLLM takes the call (at
    most 9 tokens for Kimi-K3's 896 bf16 experts), else aiter's GEMM, which AKLR's aiter provider answers."""
    reason = _keeps_vllms(hidden_states, router_weight)
    if reason is not None:
        return reason
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(hidden_states, router_weight, None, torch.float32)


@vllm_route(_CALLER, f"{_ROUTER}.can_use_rocm_fp32_router_gemm", versions=_VERSIONS)
def _can_use_rocm_fp32_router_gemm(hidden_states, router_weight):
    """Whether vLLM takes its fp32 router GEMM: vLLM's own answer where it takes the call itself, else yes for
    the bf16 gate, which ``_rocm_fp32_router_gemm`` then hands to aiter's GEMM. For Kimi-K3's router vLLM's
    kernel stops at 9 tokens (hipBLASLt from 10, its decode's 16 tokens included)."""
    reason = _keeps_vllms(hidden_states, router_weight)
    return True if reason is None else reason


def _keeps_vllms(hidden_states, router_weight) -> FallbackReason | None:
    """Why vLLM keeps its own answer and kernel for the call; ``None`` where the call is aiter's GEMM's."""
    if _native_can_use(hidden_states, router_weight):
        return FallbackReason.NOT_QUALIFIED  # vLLM's own answer: yes
    # vLLM runs an fp32 router weight through F.linear on fp32 rows, which is no 16-bit GEMM
    if hidden_states.dtype != torch.bfloat16 or router_weight.dtype != torch.bfloat16:
        return FallbackReason.OUT_OF_CONTRACT
    return None


def _native_can_use(hidden_states, router_weight) -> bool:
    """vLLM's own answer: whether its fp32 router kernel takes the call."""
    return _original(_can_use_rocm_fp32_router_gemm)(hidden_states, router_weight)
