# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.minimax_m2.contract import (
    MINIMAX_M2_FP8_MOE,
    MINIMAX_M2_MXFP4_MOE,
)

__all__ = ["minimax_m2_fp8_moe", "minimax_m2_mxfp4_moe"]


@api(MINIMAX_M2_FP8_MOE)
def minimax_m2_fp8_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    expert_offset=0,
): ...


@api(MINIMAX_M2_MXFP4_MOE)
def minimax_m2_mxfp4_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    expert_offset=0,
): ...


@sglang_route("sglang.srt.models.minimax_m2.MiniMaxM2MoE.forward_normal", versions=("0.5.22",))
def _moe_forward_normal(self, hidden_states):
    """The MoE's gate, routing and experts on this rank as one op call; sglang's layer
    boundary sums the ranks' shares after it, as it does sglang's own."""
    if moe_blocks.runtime_falls_back() or _moe_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    op, arguments = _moe_call(self)
    return op(hidden_states=hidden_states, **arguments)


def _moe_falls_back(moe) -> bool:
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    return (
        # the experts would take the rows gathered over the attention ranks, not this rank's
        is_dp_attention_enabled()
        or moe_blocks.topk_falls_back(moe.topk, scoring_func="sigmoid", biased=True)
        or moe_blocks.experts_fall_back(moe.experts, ("Fp8MoEMethod", "QuarkW4A4MXFp4MoE"))
    )


def _moe_call(moe):
    """The op for the experts' format, and its arguments but the input; ``expert_offset`` is the
    router row of this rank's first expert."""
    experts, routing = moe.experts, moe.topk.topk_config
    arguments = {
        "router_weight": moe.gate.weight,
        "correction_bias": routing.correction_bias,
        "w1": experts.w13_weight,
        "w2": experts.w2_weight,
        "topk": routing.top_k,
        "expert_offset": experts.moe_ep_rank * experts._num_local_routed,
    }
    if moe_blocks.experts_format(experts) == "Fp8MoEMethod":
        return minimax_m2_fp8_moe, {
            **arguments,
            "w1_scale": experts.w13_weight_scale_inv,
            "w2_scale": experts.w2_weight_scale_inv,
        }
    return minimax_m2_mxfp4_moe, {
        **arguments,
        "w1_scale": experts.w13_weight_scale,
        "w2_scale": experts.w2_weight_scale,
    }
