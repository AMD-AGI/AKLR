###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["biased_grouped_topk_reference"]


def biased_grouped_topk_reference(
    router_logits,
    correction_bias,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    routed_scaling_factor=1.0,
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids): each token's topk experts by biased score, within
    the topk_group groups whose two best biased scores sum highest."""
    import torch

    scores = router_logits.double().sigmoid()
    biased = scores + correction_bias.double()
    groups = biased.unflatten(-1, (num_expert_group, -1))
    group_scores = groups.topk(min(2, groups.shape[-1]), dim=-1).values.sum(dim=-1)
    kept = group_scores.topk(topk_group, dim=-1).indices
    dropped = torch.ones_like(group_scores, dtype=torch.bool).scatter(-1, kept, False)
    biased = groups.masked_fill(dropped.unsqueeze(-1), -torch.inf).flatten(-2)
    chosen = biased.topk(topk, dim=-1).indices
    weights = scores.gather(-1, chosen)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * routed_scaling_factor, chosen.double()
