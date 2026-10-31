# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["topk_transform_paged_reference"]


def topk_transform_paged_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
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
