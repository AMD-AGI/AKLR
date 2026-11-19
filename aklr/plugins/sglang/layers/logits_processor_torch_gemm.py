# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The lm_head GEMM sglang runs on torch for an unquantized head (``srt/layers/logits_processor.py``):
``hidden_states @ lm_head.weight.T``, over the rank's slice of the vocabulary."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route, sglang_version

__all__: list[str] = []


@sglang_route(
    "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
    versions=("0.5.23", "0.5.23.dev20261111"),
)
def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
    import importlib

    sglang = importlib.import_module("sglang.srt.layers.logits_processor")
    # LoRA, a quantized head, logits in fp32 and the other branches sglang takes before its plain
    # matmul, which ignores embedding_bias; the nightly's call of aiter's GEMM has its own route
    if (
        (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or sglang.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
        or not hasattr(lm_head, "weight")
        or self.use_fp32_lm_head
        or self.use_fp32_lm_head_output
        or sglang.use_intel_amx_backend(lm_head)
        or self.rl_on_policy_target is not None
        or (hasattr(sglang, "_aiter_lm_head_ok") and sglang._aiter_lm_head_ok(hidden_states, lm_head.weight))
    ):
        return FallbackReason.OUT_OF_CONTRACT
    # 0.5.23's matmul writes no buffer; the nightly's writes one of the logits' shape and the head's dtype
    if out is not None and (
        sglang_version() == "0.5.23"
        or out.shape != (hidden_states.shape[0], lm_head.weight.shape[0])
        or out.dtype != lm_head.weight.dtype
    ):
        out = None
    # hidden states of another dtype than the head's go back to sglang, which converts them: none is copied first
    return gemm(hidden_states, lm_head.weight, out=out)
