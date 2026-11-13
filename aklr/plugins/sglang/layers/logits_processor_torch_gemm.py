# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The lm_head GEMM sglang runs on torch for an unquantized head (``srt/layers/logits_processor.py``):
``hidden_states @ lm_head.weight.T``, over the rank's slice of the vocabulary."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head", versions=("0.5.22",))
def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
    from sglang.srt.utils.common import use_intel_amx_backend

    # LoRA, a quantized head, logits in fp32 and the other branches sglang takes before its
    # plain matmul, which ignores embedding_bias
    if (
        (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
        or not hasattr(lm_head, "weight")
        or self.use_fp32_lm_head
        or use_intel_amx_backend(lm_head)
        or self.rl_on_policy_target is not None
    ):
        return FallbackReason.OUT_OF_CONTRACT
    # hidden states of another dtype than the head's go back to sglang, which converts them: none is copied first
    return gemm(hidden_states, lm_head.weight)
