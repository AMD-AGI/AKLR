# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
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
    "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts",
    versions=">=0.5.22,<0.5.23",
)
def _forward_router_experts(self, hidden_states, *, defer_finalize=False):
    """The block's router and experts, its shared expert fused into the experts' last slot,
    as one op call (qwen2_moe_fp8_router_experts); where that op does not serve, the
    router op, then the experts op on its routing, or sglang's experts with the shared
    expert appended as sglang appends it. The output is this rank's share, as sglang's:
    the block's caller sums it over the TP group. Every op returns its reason before
    anything is written, and sglang's whole method runs when the router op does not serve.
    """
    import torch

    if defer_finalize or _block_falls_back(self, hidden_states):
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
        StandardTopKOutput(topk_weights, topk_ids, None), hidden_states
    )
    return self.experts(hidden_states, topk_output)


def _block_falls_back(block, hidden_states) -> bool:
    """When the block's _forward_router_experts leaves the ops: hidden states that are no
    tensor (a quantized tuple), the shared expert not fused into the experts' single last
    slot or its gate with a bias, expert parallelism, routing off its plain path
    (_routing_falls_back), experts off fp8 128 x 128 blocks on aiter's fused_moe or reducing
    their own output (_experts_fall_back), and a forward sglang compiles or splits around
    the experts (torch.compile, a piecewise or breakable CUDA graph)."""
    import torch
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.runtime_context import get_parallel

    return (
        not torch.is_tensor(hidden_states)
        or not block.enable_shared_expert_fusion
        or block.num_fused_shared_experts != 1
        or block.shared_expert is not None
        or getattr(block.shared_expert_gate, "bias", None) is not None
        or get_parallel().moe_ep_size != 1
        or _routing_falls_back(block)
        or _experts_fall_back(block.experts)
        or torch.compiler.is_compiling()
        or is_in_tc_piecewise_cuda_graph()
        or is_in_breakable_cuda_graph()
    )


def _routing_falls_back(block) -> bool:
    """When the block's TopK leaves its plain softmax top-k with renormalize: grouped
    top-k, a custom routing function, a correction bias, another score, no
    renormalize, a scaling factor, waterfill balancing, deterministic inference, and
    anything that records, remaps or overrides the routing."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    config, exec_ = block.topk.topk_config, get_exec()
    return (
        config.use_grouped_topk
        or config.custom_routing_function is not None
        or config.correction_bias is not None
        or config.scoring_func != "softmax"
        or not config.renormalize
        or config.num_fused_shared_experts != 0
        or config.routed_scaling_factor not in (None, 1.0)
        or block.topk.enable_waterfill
        or exec_.deterministic.enable_deterministic_inference
        or exec_.moe.enable_eplb
        or exec_.moe.init_expert_location != "trivial"
        or exec_.moe.ep_num_redundant_experts > 0
        or exec_.moe.expert_distribution_recorder_mode is not None
        or exec_.features.enable_return_routed_experts
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def _experts_fall_back(experts) -> bool:
    """When the block's FusedMoE experts run other than as the experts ops do: not
    Fp8MoEMethod's fp8 weights at a scale per 128 x 128 block (MXFP8, fp4 experts,
    per-channel scales, a LoRA wrapper), off aiter's runner, an activation other than gated
    silu, the routing weights on the inputs, no combine, gemm1's alpha or clamp, a SwiGLU
    limit, expert biases, padding or an expert mask, experts that reduce their own
    output, and weights DWDP prefetches."""
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    method = experts.quant_method
    if type(method) is not Fp8MoEMethod:
        return True
    runner = getattr(method, "runner", None)
    if runner is None or not runner.runner_backend.is_aiter():
        return True
    config = runner.config
    return (
        not method.block_quant
        or method.use_mxfp8
        or method.is_fp4_expert
        or list(method.weight_block_size) != [128, 128]
        or config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or config.gemm1_alpha is not None
        or config.gemm1_clamp_limit is not None
        or config.swiglu_limit is not None
        or getattr(experts, "w13_weight_bias", None) is not None
        or getattr(experts, "w2_weight_bias", None) is not None
        or getattr(experts, "hidden_pad", 0)
        or getattr(experts, "intermediate_pad", 0)
        or experts.dispatcher.expert_mask_gpu is not None
        or experts.reduce_results
        or experts._dwdp_bound
    )
