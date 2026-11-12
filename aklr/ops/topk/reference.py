###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["biased_grouped_topk_reference", "topk_transform_paged_reference"]


def topk_transform_paged_reference(scores, seq_lens, page_tables=None, *, k, page_size=1) -> Tensor:
    """Each row's selected candidates, mapped through its page table when given."""
    import torch

    width = min(k, scores.shape[-1])
    lengths = seq_lens.long().unsqueeze(-1)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    # float32 compares exactly: ranking needs no decode.
    ranked = scores.masked_fill(positions >= lengths, -torch.inf).argsort(
        dim=-1, descending=True, stable=True
    )[:, :width]
    selected = torch.where(lengths <= k, positions[:width], ranked)
    is_selected = positions[:width] < lengths
    if page_tables is not None:
        rows = is_selected.nonzero(as_tuple=True)[0]
        chosen = selected[is_selected]
        pages = page_tables[rows, chosen // page_size].long()
        selected = selected.masked_scatter(is_selected, pages * page_size + chosen % page_size)
    return torch.nn.functional.pad(torch.where(is_selected, selected, -1), (0, k - width), value=-1).double()


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
    # torch.topk orders ties as its algorithm happens to: a stable sort puts the lower expert first
    chosen = biased.sort(dim=-1, descending=True, stable=True).indices[..., :topk]
    weights = scores.gather(-1, chosen)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * routed_scaling_factor, chosen.double()
