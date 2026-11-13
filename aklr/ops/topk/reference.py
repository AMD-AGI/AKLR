# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["biased_grouped_topk_reference", "topk_transform_paged_reference"]


def biased_grouped_topk_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids): each token's topk experts by biased score, ties to the lower expert.

    Every group is kept (the contract's constraint), so the groups mask nothing.
    """
    # scored in float64: float32 ties distinct scores, and the ids compare exactly
    logits = decode(row_major_args["router_logits"], point.dtypes["router_logits"]).double()
    bias = decode(row_major_args["correction_bias"], point.dtypes["correction_bias"]).double()
    scores = logits.sigmoid()
    # torch.topk orders equal values as its algorithm happens to: a stable sort fixes them
    ranked = (scores + bias).sort(dim=-1, descending=True, stable=True).indices
    chosen = ranked[..., : point.axes["topk"]]
    weights = scores.gather(-1, chosen)
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * row_major_args["routed_scaling_factor"], chosen.double()


def topk_transform_paged_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each row's selected candidates, mapped through its page table when given."""
    import torch

    scores = row_major_args["scores"]
    page_tables = row_major_args.get("page_tables")
    topk = point.axes["topk"]
    width = min(topk, scores.shape[-1])
    lengths = row_major_args["seq_lens"].long().unsqueeze(-1)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    # float32 compares exactly: ranking needs no decode.
    ranked = scores.masked_fill(positions >= lengths, -torch.inf).argsort(
        dim=-1, descending=True, stable=True
    )[:, :width]
    selected = torch.where(lengths <= topk, positions[:width], ranked)
    is_selected = positions[:width] < lengths
    if page_tables is not None:
        page_size = point.attrs["page_size"]
        rows = is_selected.nonzero(as_tuple=True)[0]
        chosen = selected[is_selected]
        pages = page_tables[rows, chosen // page_size].long()
        selected = selected.masked_scatter(is_selected, pages * page_size + chosen % page_size)
    return torch.nn.functional.pad(
        torch.where(is_selected, selected, -1), (0, topk - width), value=-1
    ).double()
