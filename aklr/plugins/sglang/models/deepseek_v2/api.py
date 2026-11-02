# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.deepseek_v2.contract import DEEPSEEK_V2_MXFP4_MOE

__all__ = ["deepseek_v2_mxfp4_moe"]


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


@sglang_route("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal", versions=("0.5.22",))
def _moe_forward_normal(
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    input_ids=None,
    input_ids_global=None,
    skip_shared_experts=False,
    num_token_non_padded=None,
    use_vision_topk=False,
    return_moe_output=False,
):
    """The MoE's router, top-k and experts, the shared one last, as one op call; then
    forward_normal's all-reduce unless a later step sums the ranks."""
    if _forward_normal_falls_back(
        self, skip_shared_experts, num_token_non_padded, use_vision_topk, return_moe_output
    ):
        return FallbackReason.OUT_OF_CONTRACT
    out = deepseek_v2_mxfp4_moe(**_moe_arguments(self, hidden_states))
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)


def _forward_normal_falls_back(
    mlp, skip_shared_experts, num_token_non_padded, use_vision_topk, return_moe_output
) -> bool:
    # a rank's share of the experts (expert parallelism) or the routed ones alone (the shared
    # expert apart) are another shape than the op's w1, which holds the shared one last
    return (
        skip_shared_experts
        or num_token_non_padded is not None
        or use_vision_topk
        or return_moe_output
        or moe_blocks.runtime_falls_back()
        or moe_blocks.experts_fall_back(mlp.experts, ("QuarkW4A4MXFp4MoE",))
        or mlp._fuse_shared_experts_inside_sbo
        or mlp.is_hash
        or mlp.gate.e_score_correction_bias_vl is not None
    )


def _moe_arguments(mlp, hidden_states):
    experts, routing = mlp.experts, mlp.topk.topk_config
    grouped = routing.use_grouped_topk
    return {
        "hidden_states": hidden_states,
        "router_weight": mlp.gate.weight,
        "correction_bias": routing.correction_bias,
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale,
        "topk": routing.top_k - routing.num_fused_shared_experts,
        "num_expert_group": routing.num_expert_group if grouped else 1,
        "topk_group": routing.topk_group if grouped else 1,
        "renormalize": routing.renormalize,
        "scoring_func": routing.scoring_func,
        "routed_scaling_factor": mlp.routed_scaling_factor,
    }
