# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.layers.fused_topk_bias.contract import DEEPSEEK_V4_TOPK

__all__ = ["deepseek_v4_topk"]

_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"


@api(DEEPSEEK_V4_TOPK)
def deepseek_v4_topk(
    router_logits,
    correction_bias,
    topk_weights,
    topk_ids,
    renormalize,
    padding_mask=None,
    input_tokens=None,
    bias_vl=None,
    image_sentinel_lo=0,
    routed_scaling_factor=1.0,
): ...


@vllm_route(f"{_MODULE}.fused_topk_bias", f"{_MODULE}.vllm_topk_softplus_sqrt", versions=("0.34.0",))
def _vllm_topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    input_tokens=None,
    hash_indices_table=None,
    routed_scaling_factor=1.0,
    bias_vl=None,
    image_sentinel_lo=0,
):
    """DeepSeek-V4's sqrtsoftplus top-k with aiter's fused MoE on, into vLLM's buffers, with the padded rows
    VLLM_MOE_SKIP_PADDING marks and DeepSeek-V4.1's image sentinels; the hash-routed layers stay on vLLM.
    token_expert_indices is vLLM's scratch: fused_topk_bias returns the weights and ids only."""
    import torch
    from vllm.model_executor.layers.fused_moe.router.base_router import get_moe_padding_mask

    # vLLM's image_sentinel_lo 0 leaves bias_vl unread
    vision = bias_vl is not None and image_sentinel_lo != 0
    if (
        hash_indices_table is not None
        or (vision and input_tokens is None)
        or e_score_correction_bias is None
        or topk_indices.dtype != torch.int32
    ):
        return FallbackReason.OUT_OF_CONTRACT
    # (topk_weights, topk_indices) as vLLM's returns them, or why vLLM keeps the call
    return deepseek_v4_topk(
        router_logits=gating_output,
        correction_bias=e_score_correction_bias,
        topk_weights=topk_weights,
        topk_ids=topk_indices,
        renormalize=renormalize,
        padding_mask=get_moe_padding_mask(topk_indices.shape[0]),
        input_tokens=input_tokens if vision else None,
        bias_vl=bias_vl if vision else None,
        image_sentinel_lo=image_sentinel_lo if vision else 0,
        routed_scaling_factor=routed_scaling_factor,
    )
