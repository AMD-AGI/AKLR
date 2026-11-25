# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["DeepseekV4RouterReference"]


class DeepseekV4RouterReference(OpReference):
    def reference(
        self,
        *,
        router_logits,
        correction_bias,
        topk_ids,
        n_shared,
        renormalize,
        num_token_non_padded=None,
        routed_scaling_factor=1.0,
        shared_scale=1.0,
        **_,
    ):
        """topk_weights and topk_ids: each token's topk routed experts by biased sqrtsoftplus score,
        then the shared experts; padded rows keep only the shared ids.

        Rounding points of the op: the score sqrt(softplus(x)) and the biased score are float32, as
        aiter's topk_gating computes them (+inf clamped to FLT_MAX, softplus(x) = x past 20), and equal
        biased scores rank as aiter's wave64 kernel ranks them. Lane l holds experts l, l + 64, ...;
        each lane orders its experts with aiter's sorting network (compare-and-swap only on a strictly
        smaller value, so equal values keep the network's order, not the ids'), and the lanes' lists
        merge by value, the lower lane first on equal values. Distinct bf16 logits can round to one
        float32 score, so the ranking is part of the op. The renormalization divides by the weights'
        sum floored at 1e-20, as the kernel does, in float32 like the kernel.
        """
        n_experts, topk = router_logits.shape[1], topk_ids.shape[-1] - n_shared
        x = router_logits.float().clamp(max=torch.finfo(torch.float32).max)
        softplus = torch.where(
            x > 20.0, x, torch.log2(1.0 + torch.exp2(x * 1.4426950408889634)) * 0.6931471805599453
        )
        scores = softplus.sqrt()
        chosen = _aiter_topk_order(scores + correction_bias.float(), topk)
        weights = scores.gather(-1, chosen)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        weights = weights * routed_scaling_factor

        n_tokens = x.shape[0]
        shared_ids = torch.arange(n_experts, n_experts + n_shared, dtype=chosen.dtype, device=x.device)
        ids = torch.cat([chosen, shared_ids.expand(n_tokens, n_shared)], dim=-1)
        shared_weights = torch.full((n_tokens, n_shared), shared_scale, dtype=torch.float32, device=x.device)
        weights = torch.cat([weights, shared_weights], dim=-1)

        if num_token_non_padded is not None:
            n_real = int(num_token_non_padded.reshape(-1)[0])
            padded = torch.arange(n_tokens, device=x.device) >= n_real
            ids[padded, :topk] = 0
            weights[padded] = 0
        return {"topk_weights": weights, "topk_ids": ids.to(topk_ids.dtype)}


# aiter's sort_network_desc<EPT> comparator pairs, by experts per lane (64, 128, 256, 384 experts)
_SORT_NETWORK = {
    1: (),
    2: ((0, 1),),
    4: ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)),
    6: ((0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (2, 3)),
}


def _aiter_topk_order(biased, topk):
    """The ids of each row's topk largest biased scores, in aiter's order (the reference's docstring)."""
    n_rows, n_experts = biased.shape
    per_lane = n_experts // 64
    # [rows, lane, slot]: lane l, slot j holds expert l + 64 * j
    values = biased.view(n_rows, per_lane, 64).transpose(1, 2).clone()
    ids = (
        torch.arange(n_experts, device=biased.device)
        .view(per_lane, 64)
        .t()
        .expand(n_rows, 64, per_lane)
        .clone()
    )
    for i, j in _SORT_NETWORK[per_lane]:
        swap = values[..., i] < values[..., j]
        vi, vj = values[..., i].clone(), values[..., j].clone()
        ii, ij = ids[..., i].clone(), ids[..., j].clone()
        values[..., i], values[..., j] = torch.where(swap, vj, vi), torch.where(swap, vi, vj)
        ids[..., i], ids[..., j] = torch.where(swap, ij, ii), torch.where(swap, ii, ij)
    # flattened lane-major: a stable descending sort by value keeps the lower lane, then the
    # earlier slot of its sorted list, first on equal values
    order = values.reshape(n_rows, -1).sort(dim=-1, descending=True, stable=True).indices[:, :topk]
    return ids.reshape(n_rows, -1).gather(-1, order)
