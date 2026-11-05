# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.qwen3_moe.contract import QWEN3_MOE_FP8_MOE

__all__ = ["qwen3_moe_fp8_moe"]


@api(QWEN3_MOE_FP8_MOE)
def qwen3_moe_fp8_moe(
    hidden_states,
    router_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=8,
): ...


@sglang_route("sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal", versions=("0.5.22",))
def _moe_forward_normal(self, hidden_states):
    """The block's gate, top-k and experts on this rank as one op call; sglang's layer
    boundary sums the ranks' shares after it, as it does sglang's own."""
    if moe_blocks.runtime_falls_back() or _block_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    experts = self.experts
    return qwen3_moe_fp8_moe(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale_inv,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale_inv,
        topk=self.topk.topk_config.top_k,
    )


def _block_falls_back(block) -> bool:
    from sglang.srt.runtime_context import get_parallel

    gate = block.gate
    return (
        # the op indexes w1 by the router's ids over every expert, not this rank's
        get_parallel().moe_ep_size != 1
        # a LoRA wrapper, or a gate the checkpoint quantizes (ModelSlim)
        or type(gate).__name__ != "ReplicatedLinear"
        or type(gate.quant_method).__name__ != "UnquantizedLinearMethod"
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("Fp8MoEMethod",))
    )
