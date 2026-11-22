# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.deepseek_v2.contract import (
    DEEPSEEK_V2_FP8_MOE,
    DEEPSEEK_V2_MXFP4_MOE,
)

__all__ = ["deepseek_v2_fp8_moe", "deepseek_v2_mxfp4_moe"]


@api(DEEPSEEK_V2_MXFP4_MOE)
def deepseek_v2_mxfp4_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@api(DEEPSEEK_V2_FP8_MOE)
def deepseek_v2_fp8_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@sglang_route(
    "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal",
    versions=("0.5.23", "0.5.23.dev20261120"),
)
def _moe_forward_normal(
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    input_ids=None,
    input_ids_global=None,
    skip_shared_experts=False,
    use_vision_topk=False,
    return_moe_output=False,
    leave_output_sum=False,
):
    """The MoE's router, top-k and experts, the shared one last, as one op call; then
    forward_normal's all-reduce unless a later step sums the ranks."""
    if _forward_normal_falls_back(
        self, skip_shared_experts, use_vision_topk, return_moe_output or leave_output_sum
    ):
        return FallbackReason.OUT_OF_CONTRACT
    op, arguments = _moe_call(self, hidden_states)
    out = op(**arguments)
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)


# the experts' formats the ops take: Quark's MXFP4, fp8 at a scale per 128 x 128 block
_EXPERT_FORMATS = ("QuarkW4A4MXFp4MoE", "Fp8MoEMethod")


def _forward_normal_falls_back(mlp, skip_shared_experts, use_vision_topk, return_moe_output) -> bool:
    # a rank's share of the experts (expert parallelism) or the routed ones alone (the shared
    # expert apart) are another shape than the op's w1, which holds the shared one last
    return (
        skip_shared_experts
        or use_vision_topk
        or return_moe_output
        or _masks_padded_rows()
        or moe_blocks.runtime_falls_back()
        or moe_blocks.experts_fall_back(mlp.experts, _EXPERT_FORMATS)
        or mlp._fuse_shared_experts_inside_sbo
        or mlp.is_hash
        or mlp.gate.e_score_correction_bias_vl is not None
    )


def _masks_padded_rows() -> bool:
    """Whether sglang's top-k masks padded rows here: the bound it reads from the forward context
    (resolve_routing_inputs), forward_normal's num_token_non_padded until 0.5.23."""
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context

    if not has_forward_context():
        return False
    get_bound = get_forward_context().moe_num_token_non_padded
    return get_bound is not None and get_bound() is not None


def _moe_call(mlp, hidden_states):
    """The op for the experts' format, and its arguments: the module's weights and input."""
    experts, routing = mlp.experts, mlp.topk.topk_config
    grouped = routing.use_grouped_topk
    arguments = {
        "hidden_states": hidden_states,
        "router_weight": mlp.gate.weight,
        "correction_bias": routing.correction_bias,
        "w1": experts.w13_weight,
        "w2": experts.w2_weight,
        "topk": routing.top_k - moe_blocks.fused_shared_experts(routing),
        "num_expert_group": routing.num_expert_group if grouped else 1,
        "topk_group": routing.topk_group if grouped else 1,
        "renormalize": routing.renormalize,
        "scoring_func": routing.scoring_func,
        "routed_scaling_factor": mlp.routed_scaling_factor,
    }
    if moe_blocks.experts_format(experts) == "Fp8MoEMethod":
        return deepseek_v2_fp8_moe, {
            **arguments,
            "correction_bias": _topk_bias(routing, hidden_states.dtype),
            "w1_scale": experts.w13_weight_scale_inv,
            "w2_scale": experts.w2_weight_scale_inv,
        }
    return deepseek_v2_mxfp4_moe, {
        **arguments,
        "w1_scale": experts.w13_weight_scale,
        "w2_scale": experts.w2_weight_scale,
    }


def _topk_bias(routing, logits_dtype):
    """``routing``'s correction bias as sglang's select_experts hands it to the top-k: in the
    router logits' dtype to aiter's grouped top-k (the copy TopKConfig keeps), else as held.
    GLM-5.2 holds its bias in fp32, which aiter's top-k reads in bf16."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.topk import _use_aiter

    if _use_aiter and routing.use_grouped_topk and not envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.get():
        return routing.correction_bias_for_dtype(logits_dtype)
    return routing.correction_bias
