# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["mega_moe_baseline"]

# sglang's DeepSeek-V4 MegaMoE capacity (SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR)
_MAX_TOKENS_PER_RANK = 8192


# one aiter instance per device and shape, as sglang keeps one for every layer
_INSTANCES: dict = {}


def _instance(key, weights):
    """aiter's MegaMoEV2 of this group, built at its first call."""
    import torch
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    if key in _INSTANCES:
        return _INSTANCES[key]
    _, model_dim, inter_dim, n_experts, topk, ep_size, swiglu_limit = key
    rank = torch.distributed.get_rank() if ep_size > 1 else 0
    w1, w1_scale, w2, w2_scale = weights
    _INSTANCES[key] = MegaMoEV2(
        rank=rank,
        world_size=ep_size,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=n_experts * ep_size,
        topk=topk,
        quant="a8w4",
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=_MAX_TOKENS_PER_RANK,
        swiglu_limit=swiglu_limit,
    )
    return _INSTANCES[key]


def mega_moe_baseline(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    ep_size=1,
    swiglu_limit=10.0,
    out_dtype=None,
    out=None,
):
    """aiter's MoE as sglang runs it: MegaMoEV2's main path over an expert-parallel group,
    this layer's weights bound before the call; with one rank, aiter's local fused_moe on
    the same gate-up-interleaved experts (MegaMoEV2 needs mori's group)."""
    import torch

    if ep_size == 1:
        from aklr.ops.fused_moe.baseline import fused_moe_baseline

        return fused_moe_baseline(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            activation="silu",
            quant_type="per_1x32",
            doweight_stage1=False,
            gate_mode="interleave",
            act_quant=4,
            expert_mask=None,
            num_local_tokens=None,
            beta=None,
            linear_beta=None,
            swiglu_limit=swiglu_limit,
            out_dtype=out_dtype,
            out=None,
        )
    weights = tuple(t.contiguous().view(torch.uint8) for t in (w1, w1_scale, w2, w2_scale))
    key = (
        hidden_states.device,
        hidden_states.shape[1],
        w1.shape[1] // 2,
        w1.shape[0],
        topk_ids.shape[1],
        ep_size,
        float(swiglu_limit),
    )
    mega = _instance(key, weights)
    mega._s1_w1, mega._s1_w1_scale, mega.w2, mega.w2_scale = weights
    out = mega.forward(hidden_states, topk_weights, topk_ids, slice_output=True)
    return out.clone() if out_dtype is None else out.to(out_dtype)
