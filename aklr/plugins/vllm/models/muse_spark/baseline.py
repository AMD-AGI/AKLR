# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM 0.33.0rc1.dev910 runs on aiter on gfx950 from MuseSpark1xMoE._pool's norm to
the experts' per-route output: fused_rmsnorm_expand, then MuseSparkAiterExperts.apply's
routes paired with the always-masked slot and its cached unit weights, into aiter's
fused_moe at per_1x32 (rocm_aiter_fused_experts' arguments for MXFP4 w4a4 experts).
Without vLLM: its norm kernel is copied in ``_norm_expand``."""

from __future__ import annotations

import functools
import os

__all__ = ["muse_spark_moe_baseline"]


@functools.cache
def _unit_weights(device, n_routes: int):
    """MuseSparkAiterExperts._unit_weights: ones, built once per size."""
    import torch

    return torch.ones((n_routes, 2), dtype=torch.float32, device=device)


def muse_spark_moe_baseline(
    hidden_states,
    norm_weight,
    topk_ids,
    expert_mask,
    w1,
    w1_scale,
    w2,
    w2_scale,
    eps=1e-6,
    gain_center=1.0,
    out_dtype=None,
    out=None,
):
    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    from aklr.plugins.vllm.models.muse_spark._norm_expand import rmsnorm_expand

    n_tokens, topk = topk_ids.shape
    expanded = rmsnorm_expand(hidden_states, norm_weight, eps, gain_center, False, topk)
    flat_ids = topk_ids.view(n_tokens * topk, 1)
    masked_slot = flat_ids.new_full((n_tokens * topk, 1), expert_mask.shape[0] - 1)
    route_output = fused_moe(
        expanded,
        w1,
        w2,
        _unit_weights(topk_ids.device, n_tokens * topk),
        torch.cat((flat_ids, masked_slot), dim=1),
        expert_mask=expert_mask,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        # rocm_aiter_ops.get_moe_dispatch_policy(): vLLM's env, default 0
        moe_sorting_dispatch_policy=int(os.environ.get("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "0")),
        dtype=hidden_states.dtype,
        gate_mode=GateMode.SEPARATED.value,
    )
    return route_output.view(n_tokens, topk, -1)
