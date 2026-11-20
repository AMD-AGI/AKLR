# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE routing on ROCm (``model_executor/layers/fused_moe/router``): the gate GEMM with an fp32
output that gfx950 runs for a few tokens (Kimi-K3's router)."""

from __future__ import annotations

import os

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []

_VERSIONS = ("0.34.0", "0.32.1rc1.dev276")


@vllm_route(
    "vllm.model_executor.layers.fused_moe.router.gate_linear.fp32_router_gemm_dispatch_impl",
    "vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm.rocm_fp32_router_gemm",
    versions=_VERSIONS,
)
def _rocm_fp32_router_gemm(hidden_states, router_weight):
    """The gate's ``hidden_states @ router_weight.T`` in fp32, which vLLM runs on gfx950 for at most 32
    tokens (at most 9 for Kimi-K3's 896 bf16 experts, unless AKLR serves the call: below)."""
    return gemm(hidden_states, router_weight, out_dtype=torch.float32)


@vllm_route(
    "vllm.model_executor.layers.fused_moe.router.gate_linear.fp32_router_gemm_dispatch_impl",
    "vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm.can_use_rocm_fp32_router_gemm",
    versions=_VERSIONS,
)
def _can_use_rocm_fp32_router_gemm(hidden_states, router_weight):
    """Whether vLLM takes its fp32 router GEMM, whose call reaches AKLR above: yes where AKLR serves it, else
    vLLM's own answer. vLLM's kernel stops at 9 tokens for Kimi-K3's router (hipBLASLt from there, its decode's
    16 tokens included); AKLR's serves the counts its workloads measured faster than that."""
    key = (
        tuple(hidden_states.shape), hidden_states.stride(), hidden_states.dtype, hidden_states.device,
        tuple(router_weight.shape), router_weight.stride(), router_weight.dtype,
    )  # fmt: skip
    served = _SERVED.get(key)
    if served is None:
        # one dispatch lookup per call shape: vLLM asks on every MoE layer, every step
        served = gemm.fallback_reason(hidden_states, router_weight, out_dtype=torch.float32) is None
        if _REMEMBER:
            _SERVED[key] = served
    return True if served else FallbackReason.NOT_MEASURED


# the answer per call shape; not kept while exporting, which counts every call
_SERVED: dict[tuple, bool] = {}
_REMEMBER = os.environ.get("AKLR_EXPORT") != "1"
