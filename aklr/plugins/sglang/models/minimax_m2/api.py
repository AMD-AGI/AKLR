# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
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


@sglang_route(
    "sglang.srt.models.minimax_m2.MiniMaxM2MoE.forward_normal",
    versions=">=0.5.22,<0.5.23",
)
def _moe_forward_normal(self, hidden_states):
    """The MoE of a MiniMaxM2DecoderLayer as one op call: its gate, routing and experts on
    this rank, returned as forward_normal returns them, this rank's share; sglang's layer
    boundary sums the shares over the ranks after it (ffn_boundary.finish), as it does
    sglang's own. Falls back for what the ops do not express (_moe_falls_back); the contract
    checks shapes, dtypes and the weights' layout."""
    if hidden_states.shape[0] == 0 or _moe_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    op, arguments = _moe_call(self)
    return op(hidden_states=hidden_states, **arguments)


def _moe_falls_back(moe) -> bool:
    """When MiniMaxM2MoE leaves its ops: routing off its plain path (_routing_falls_back),
    experts off sglang's standard dispatch on one TP group (_dispatch_falls_back), or neither
    fp8 at 128 x 128 blocks nor Quark's MXFP4 on aiter's fused_moe (_fp8_experts_fall_back,
    _mxfp4_experts_fall_back)."""
    experts = moe.experts
    if _routing_falls_back(moe) or _dispatch_falls_back(moe):
        return True
    if _experts_format(experts) == "Fp8MoEMethod":
        return _fp8_experts_fall_back(experts)
    if _experts_format(experts) == "QuarkW4A4MXFp4MoE":
        return _mxfp4_experts_fall_back(experts)
    return True


def _routing_falls_back(moe) -> bool:
    """When the routing leaves the ops' plain path: a gate with a bias, no correction bias,
    scores other than renormalized sigmoid ones at scale 1, grouped or custom top-k, fused
    shared experts, deterministic inference, and anything that records, remaps or overrides
    the routing."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    routing, exec_ = moe.topk.topk_config, get_exec()
    return (
        getattr(moe.gate, "bias", None) is not None
        or routing.correction_bias is None
        or routing.scoring_func != "sigmoid"
        or not routing.renormalize
        or routing.routed_scaling_factor not in (None, 1.0)
        or routing.apply_routed_scaling_factor_on_output
        or routing.use_grouped_topk
        or routing.custom_routing_function is not None
        or routing.num_fused_shared_experts != 0
        or getattr(moe.topk, "waterfill_balancer", None) is not None
        or exec_.deterministic.enable_deterministic_inference
        or exec_.moe.enable_eplb
        or exec_.moe.expert_distribution_recorder_mode is not None
        or exec_.features.enable_return_routed_experts
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def _dispatch_falls_back(moe) -> bool:
    """When MiniMaxM2MoE's experts are not a FusedMoE computing, on this rank's rows, the
    router's experts it stores, as sglang's standard dispatch does: an all-to-all or DP
    attention, redundant, replicated or elastically joined experts, fused shared experts,
    or the output reduced inside."""
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec

    experts = moe.experts
    if type(experts).__name__ != "FusedMoE":  # DeepEPMoE, LoRA or KT wrappers
        return True
    return (
        not get_moe_a2a_backend().is_none()
        or type(experts.dispatcher).__name__ != "StandardDispatcher"
        or is_dp_attention_enabled()
        or experts.reduce_results
        or experts.num_fused_shared_experts != 0
        or experts.num_experts != moe.gate.weight.shape[0]
        or experts._expert_storage_rank != experts.moe_ep_rank
        or get_exec().moe.ep_join_mode is not None
        or getattr(experts, "_dwdp_bound", False)
    )


def _runner_falls_back(runner, experts) -> bool:
    """When FusedMoE ``experts`` run other than on aiter's fused_moe as the ops do: no aiter
    ``runner``, an activation other than gated silu, the routing weights on the inputs, no
    combine, gemm1's alpha or clamp, a SwiGLU limit, expert biases, or padded hidden rows.
    """
    if runner is None or not runner.runner_backend.is_aiter():
        return True
    config = runner.config
    return (
        config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or config.gemm1_alpha is not None
        or config.gemm1_clamp_limit is not None
        or config.swiglu_limit not in (None, 0.0)
        or getattr(experts, "w13_weight_bias", None) is not None
        or getattr(experts, "w2_weight_bias", None) is not None
        or getattr(experts, "hidden_pad", 0) != 0
    )


def _fp8_experts_fall_back(experts) -> bool:
    """When Fp8MoEMethod ``experts`` run other than as the fp8 op does: weights not at a scale
    per 128 x 128 block (MXFP8, fp4 experts, per-channel scales), the gate and up rows
    interleaved, a padded intermediate, or off aiter's fused_moe (_runner_falls_back).
    """
    method = experts.quant_method
    return (
        not method.block_quant
        or method.use_mxfp8
        or method.is_fp4_expert
        or list(method.weight_block_size) != [128, 128]
        or getattr(experts, "_aiter_gate_up_interleaved", False)
        or getattr(experts, "intermediate_pad", 0) != 0
        or _runner_falls_back(getattr(method, "runner", None), experts)
    )


def _mxfp4_experts_fall_back(experts) -> bool:
    """When QuarkW4A4MXFp4MoE ``experts`` run other than as the MXFP4 op does: static
    activation scales, presharded weights with a padded intermediate (the loader leaves the
    padding unwritten), or off aiter's fused_moe (_runner_falls_back). Otherwise a padded
    intermediate is in the op: sglang's loader zeroes the padding's weights, so the op's sum
    over the stored width is sglang's."""
    scheme = experts.scheme
    return (
        scheme.static_input_scales
        or (getattr(experts, "intermediate_pad", 0) != 0 and experts.use_presharded_weights)
        or _runner_falls_back(getattr(scheme, "runner", None), experts)
    )


def _experts_format(experts) -> str:
    """The class that names FusedMoE ``experts``' format, by sglang's rule: their scheme's,
    else their quant method's."""
    return type(getattr(experts, "scheme", None) or experts.quant_method).__name__


def _moe_call(moe):
    """MiniMaxM2MoE as its op and the op's arguments but its input: the gate, the routing,
    the experts as sglang holds them for aiter (shuffle_weight's 16 x 16 tiles) and the
    router row of this rank's first expert (0 without expert parallelism)."""
    experts, routing = moe.experts, moe.topk.topk_config
    arguments = {
        "router_weight": moe.gate.weight,
        "correction_bias": routing.correction_bias,
        "w1": experts.w13_weight,
        "w2": experts.w2_weight,
        "topk": routing.top_k,
        "expert_offset": experts.moe_ep_rank * experts._num_local_routed,
    }
    if _experts_format(experts) == "Fp8MoEMethod":
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
