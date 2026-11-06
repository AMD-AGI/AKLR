# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.deepseek_v4.contract import (
    DEEPSEEK_V4_ROUTER,
)

__all__ = ["deepseek_v4_router"]


@api(DEEPSEEK_V4_ROUTER)
def deepseek_v4_router(
    router_logits,
    correction_bias,
    topk_weights,
    topk_ids,
    n_shared,
    renormalize,
    num_token_non_padded=None,
    routed_scaling_factor=1.0,
    shared_scale=1.0,
): ...


@sglang_route("sglang.srt.layers.moe.topk.select_experts", versions=("0.5.23",))
def _select_experts(
    hidden_states,
    router_logits,
    topk_config,
    *,
    layer_id=None,
    num_token_non_padded=None,
    expert_location_dispatch_info=None,
    dynamic_expert_bias=None,
    router_logits_partials=None,
):
    """The routed top-k and the shared columns as one op call, returned as sglang returns them,
    for the router sglang serves on ROCm with aiter's sqrtsoftplus topk_gating and a fused
    shared expert appended globally (DeepSeek-V4); everything else stays on sglang."""
    import torch
    from sglang.srt.layers.moe import topk as sglang_topk

    if _falls_back(sglang_topk, topk_config, num_token_non_padded) or (
        expert_location_dispatch_info is not None
        or dynamic_expert_bias is not None
        # the split-K decode router (rocm_router_gate) reduces the partials first
        or router_logits_partials is not None
    ):
        return FallbackReason.OUT_OF_CONTRACT
    n_shared = topk_config.num_shared_topk_slots
    shape, device = (router_logits.shape[0], topk_config.top_k), router_logits.device
    shared_scale = topk_config.fused_shared_experts_scaling_factor
    out = deepseek_v4_router(
        router_logits=router_logits,
        correction_bias=topk_config.correction_bias,
        topk_weights=torch.empty(shape, dtype=torch.float32, device=device),
        topk_ids=torch.empty(shape, dtype=torch.int32, device=device),
        n_shared=n_shared,
        renormalize=topk_config.renormalize,
        num_token_non_padded=num_token_non_padded,
        routed_scaling_factor=topk_config.routed_scaling_factor,
        shared_scale=1.0 if shared_scale is None else shared_scale,
    )
    if isinstance(out, FallbackReason):
        return out
    topk_weights, topk_ids = out
    # sglang hands its capturer and recorder the routed columns
    routed_ids = topk_ids[:, :-n_shared]
    sglang_topk.capture_routed_experts_if_allowed(topk_config, layer_id, routed_ids)
    sglang_topk.get_global_expert_distribution_recorder().on_select_experts(topk_ids=routed_ids)
    return sglang_topk.StandardTopKOutput(topk_weights, topk_ids, router_logits)


def _falls_back(sglang_topk, config, num_token_non_padded) -> bool:
    """Whether sglang routes this call other than as the op computes it: off ROCm's aiter
    topk_gating with sqrtsoftplus, with the shared experts not appended globally after it, or
    with a later step that rewrites the ids (expert remapping, routing simulation, recorders,
    the unmasked padded rows SGLANG_MORI_NO_PAD_MASK keeps)."""
    if not (sglang_topk._use_aiter and sglang_topk._is_hip) or moe_blocks.runtime_falls_back():
        return True
    n_shared = config.num_shared_topk_slots
    return (
        config.scoring_func != "sqrtsoftplus"
        or config.use_grouped_topk
        or config.torch_native
        or config.custom_routing_function is not None
        or config.correction_bias is None
        or config.routed_scaling_factor is None
        # sqrtsoftplus_log1p and fused_gate_packed_ids need no check: sglang's aiter path with
        # fused shared experts passes neither to topk_gating
        or n_shared <= 0
        or sglang_topk.has_per_rank_fused_shared_slots(n_shared)
        or sglang_topk.is_expert_remapping_enabled()
        or (num_token_non_padded is not None and sglang_topk._skip_hip_pad_mask)
    )
