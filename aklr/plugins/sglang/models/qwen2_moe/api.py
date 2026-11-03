# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.qwen2_moe.contract import (
    QWEN2_MOE_FP8_EXPERTS,
    QWEN2_MOE_FP8_ROUTER_EXPERTS,
    QWEN2_MOE_ROUTER,
)

__all__ = [
    "qwen2_moe_fp8_experts",
    "qwen2_moe_fp8_router_experts",
    "qwen2_moe_router",
]


@api(QWEN2_MOE_ROUTER)
def qwen2_moe_router(hidden_states, router_weight, topk_weights, topk_ids): ...


@api(QWEN2_MOE_FP8_EXPERTS)
def qwen2_moe_fp8_experts(
    hidden_states,
    topk_weights,
    topk_ids,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
): ...


@api(QWEN2_MOE_FP8_ROUTER_EXPERTS)
def qwen2_moe_fp8_router_experts(
    hidden_states,
    router_weight,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=10,
): ...


@sglang_route(
    "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts", versions=("0.5.22",)
)
def _forward_router_experts(self, hidden_states, *, defer_finalize=False):
    """The router and experts, the shared expert last, as one op; where it does not serve, the
    router op, then the experts op or sglang's experts on its routing. The output is this
    rank's share, as sglang's."""
    import torch

    if (
        defer_finalize
        # a quantized tuple
        or not torch.is_tensor(hidden_states)
        or moe_blocks.runtime_falls_back()
        or _block_falls_back(self)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    experts, topk = self.experts, self.topk.topk_config.top_k
    expert_args = {
        "hidden_states": hidden_states,
        "shared_gate_weight": self.shared_expert_gate.weight,
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale_inv,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale_inv,
    }
    out = qwen2_moe_fp8_router_experts(**expert_args, router_weight=self.gate.weight, topk=topk)
    if not isinstance(out, FallbackReason):
        return out
    shape, device = (hidden_states.shape[0], topk), hidden_states.device
    routed = qwen2_moe_router(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        topk_weights=torch.empty(shape, dtype=torch.float32, device=device),
        topk_ids=torch.empty(shape, dtype=torch.int32, device=device),
    )
    if isinstance(routed, FallbackReason):
        return routed
    topk_weights, topk_ids = routed
    out = qwen2_moe_fp8_experts(**expert_args, topk_weights=topk_weights, topk_ids=topk_ids)
    if not isinstance(out, FallbackReason):
        return out
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    # as sglang's small-M router leaves it: no logits
    topk_output = self._append_shared_to_topk_output(
        StandardTopKOutput(topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None), hidden_states
    )
    return self.experts(hidden_states, topk_output)


def _block_falls_back(block) -> bool:
    from sglang.srt.runtime_context import get_parallel

    return (
        # the experts op indexes w1 by the router's ids over every expert, not this rank's
        get_parallel().moe_ep_size != 1
        or block.num_fused_shared_experts != 1
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("Fp8MoEMethod",))
    )
