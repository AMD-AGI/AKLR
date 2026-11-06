# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

__all__ = ["deepseek_v4_router_reference"]


if TYPE_CHECKING:
    from torch import Tensor


def deepseek_v4_router_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids): each token's topk routed experts by biased sqrtsoftplus score,
    then the shared experts; padded rows keep only the shared ids.

    Rounding points of the op: the score sqrt(softplus(x)) and the biased score are float32, as
    aiter's topk_gating computes them (+inf clamped to FLT_MAX, softplus(x) = x past 20), and equal
    biased scores rank as aiter's wave64 merge ranks them: by the lower lane (expert id % 64), then
    the lower expert id. Distinct bf16 logits can round to one float32
    score, so the ranking is part of the op; the weights' sum and the scaling are exact here.
    """
    import torch

    n_experts, n_shared = point.axes["n_experts"], point.attrs["n_shared"]
    topk = point.axes["n_slots"] - n_shared
    logits = decode(row_major_args["router_logits"], point.dtypes["router_logits"]).float()
    bias = decode(row_major_args["correction_bias"], point.dtypes["correction_bias"]).float()
    x = logits.clamp(max=torch.finfo(torch.float32).max)
    softplus = torch.where(
        x > 20.0, x, torch.log2(1.0 + torch.exp2(x * 1.4426950408889634)) * 0.6931471805599453
    )
    scores = softplus.sqrt()
    # experts in lane order (lane l holds l, l + 64, ...); a stable descending sort then ranks
    # equal biased scores by lane, then by id
    lane_order = torch.arange(n_experts, device=logits.device).view(-1, 64).t().reshape(-1)
    ranked = (scores + bias)[:, lane_order].sort(dim=-1, descending=True, stable=True).indices
    chosen = lane_order[ranked[:, :topk]]
    weights = scores.gather(-1, chosen).double()
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * row_major_args["routed_scaling_factor"]

    n_tokens = logits.shape[0]
    shared_ids = torch.arange(n_experts, n_experts + n_shared, dtype=torch.float64, device=logits.device)
    ids = torch.cat([chosen.double(), shared_ids.expand(n_tokens, n_shared)], dim=-1)
    shared_weights = torch.full(
        (n_tokens, n_shared), row_major_args["shared_scale"], dtype=torch.float64, device=logits.device
    )
    weights = torch.cat([weights, shared_weights], dim=-1)

    if point.attrs["has_num_token_non_padded"]:
        n_real = int(row_major_args["num_token_non_padded"].reshape(-1)[0])
        padded = torch.arange(n_tokens, device=logits.device) >= n_real
        ids[padded, :topk] = 0
        weights[padded] = 0
    return weights, ids
