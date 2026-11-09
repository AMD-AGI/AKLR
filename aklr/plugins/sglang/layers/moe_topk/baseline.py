# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["deepseek_v4_router_baseline"]


def deepseek_v4_router_baseline(
    router_logits,
    correction_bias,
    topk_weights,
    topk_ids,
    n_shared,
    renormalize,
    num_token_non_padded=None,
    routed_scaling_factor=1.0,
    shared_scale=1.0,
):
    """What sglang 0.5.23's select_experts runs on ROCm with aiter for this router: aiter's
    topk_gating (score_func sqrtsoftplus) into the routed columns, then sglang's
    fused_append_shared_experts kernel (the padded-row mask folded in), here writing the
    caller's outputs instead of new tensors."""
    import torch
    import triton
    from aiter import topk_gating
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import _fused_append_shared_experts_kernel

    n_tokens, n_slots = topk_ids.shape
    n_experts = router_logits.shape[1]
    topk = n_slots - n_shared
    routed_weights = torch.empty((n_tokens, topk), dtype=torch.float32, device=router_logits.device)
    routed_ids = torch.empty((n_tokens, topk), dtype=torch.int32, device=router_logits.device)
    topk_gating(
        routed_weights,
        routed_ids,
        router_logits,
        correction_bias,
        renormalize,
        routed_scaling_factor,
        score_func="sqrtsoftplus",
    )
    has_padding = num_token_non_padded is not None
    _fused_append_shared_experts_kernel[(n_tokens,)](
        routed_ids,
        routed_weights,
        topk_ids,
        topk_weights,
        N_BASE=n_experts,
        scale_factor=shared_scale,
        num_token_non_padded_ptr=num_token_non_padded if has_padding else routed_ids,
        K=topk,
        S=n_shared,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_S=triton.next_power_of_2(n_shared),
        HAS_PADDING=has_padding,
        num_warps=1,
    )
    return topk_weights, topk_ids
