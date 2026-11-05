# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.qwen2_moe.contract import (
    QWEN2_MOE_FP8_EXPERTS,
    QWEN2_MOE_FP8_ROUTER_EXPERTS,
    QWEN2_MOE_MXFP4_MOE,
    QWEN2_MOE_ROUTER,
)

__all__ = [
    "qwen2_moe_fp8_experts",
    "qwen2_moe_fp8_router_experts",
    "qwen2_moe_mxfp4_moe",
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


@api(QWEN2_MOE_MXFP4_MOE)
def qwen2_moe_mxfp4_moe(
    hidden_states,
    router_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_w1,
    shared_w2,
    shared_gate_weight,
    topk=10,
): ...


@sglang_route("sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward", versions=("0.5.22",))
def _forward(self, hidden_states, forward_batch=None, defer_finalize=False):
    """The whole block, its BF16 shared expert apart from the MXFP4 experts (sglang's path for
    a checkpoint that keeps its shared expert BF16), as one op; then the sum over the TP group
    where sglang's forward does it. Every other block runs sglang's forward, whose router and
    experts reach the _forward_router_experts route."""
    import torch

    if (
        defer_finalize
        or not torch.is_tensor(hidden_states)
        or hidden_states.dim() != 2
        or moe_blocks.runtime_falls_back()
        or _unfused_block_falls_back(self)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    experts, shared = self.experts, self.shared_expert
    out = qwen2_moe_mxfp4_moe(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale,
        shared_w1=shared.gate_up_proj.weight,
        shared_w2=shared.down_proj.weight,
        shared_gate_weight=self.shared_expert_gate.weight,
        topk=self.topk.topk_config.top_k,
    )
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import reduce_moe_output

    return reduce_moe_output(out)


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


def _unfused_block_falls_back(block) -> bool:
    """Whether the block runs other than as qwen2_moe_mxfp4_moe takes it: its shared expert a
    BF16 Qwen2MoeMLP apart from the experts, gated by sigmoid(hidden_states gate^T) as
    sglang's fused gate adds it, its experts Quark's MXFP4 on aiter over every routed expert."""
    import torch
    from sglang.srt.models import qwen2_moe
    from sglang.srt.runtime_context import get_forward, get_parallel

    gate = block.shared_expert_gate
    return (
        # w1 holds every routed expert, the router's ids index it
        get_parallel().moe_ep_size != 1
        # sequence parallelism gathers the shared expert's input and scatters its output
        or get_forward().sp_active
        or block.num_fused_shared_experts != 0
        or block.shared_expert is None
        or type(gate) is not torch.nn.Linear
        or gate.bias is not None
        # sglang hands the shared expert's add to its LoRA MoE dispatch
        or qwen2_moe._SGLANG_EXPERIMENTAL_LORA_OPTI
        or _bf16_linear_falls_back(block.gate, "ReplicatedLinear")
        or _shared_expert_falls_back(block.shared_expert)
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("QuarkW4A4MXFp4MoE",))
    )


def _shared_expert_falls_back(shared) -> bool:
    """Whether ``shared`` runs other than as down(silu(gate(x)) * up(x)) of its BF16 weights,
    its output this rank's share (summed over the TP group with the routed experts')."""
    up, down = shared.gate_up_proj, shared.down_proj
    return (
        type(shared).__name__ != "Qwen2MoeMLP"
        or type(shared.act_fn).__name__ != "SiluAndMul"
        or shared._enable_silu_fp4_quant_fusion
        or _bf16_linear_falls_back(up, "MergedColumnParallelLinear")
        or _bf16_linear_falls_back(down, "RowParallelLinear")
        or up.gather_output
        or not down.input_is_parallel
        # the down projection would sum the shared expert over the TP group itself
        or down.reduce_results
        or down.use_decode_attn_tp
    )


def _bf16_linear_falls_back(layer, kind: str) -> bool:
    """Whether ``layer`` is other than a plain ``kind`` without a bias on the unquantized
    method, which multiplies by its weight as stored (a LoRA layer wraps it in another
    class)."""
    return (
        type(layer).__name__ != kind
        or type(layer.quant_method).__name__ != "UnquantizedLinearMethod"
        or layer.bias is not None
    )
