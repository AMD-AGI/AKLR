# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE router gate (``GateLinear``) on ROCm, whose own dispatch never reaches aiter's tuned GEMM."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.aiter.provider.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []

_GATE = "vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear"


@vllm_route(_GATE, f"{_GATE}.forward", versions=("0.33.0rc1.dev910", "0.34.0"))
def _forward(self, x):
    """A bf16 gate's fp32 logits (vLLM's tier 6, as DeepSeek-V4's gate) as one gemm at any row
    count. Declared on forward: its kernels are picked inside a custom op, by rows (a Triton
    one to 32, torch.mm past them). The other tiers are vLLM's."""
    if not self.allow_cublas_router_gemm:
        return FallbackReason.OUT_OF_CONTRACT
    out = gemm(x, self.weight, out_dtype=self.out_dtype)
    if isinstance(out, FallbackReason) or not self.return_bias:
        return out
    return out, None
