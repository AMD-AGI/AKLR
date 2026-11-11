# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["paged_attention_decode_reference", "unified_attention_reference"]

# Gathered KV elements per chunk of requests: 256 MiB in float32, 512 MiB in float64.
_CHUNK_ELEMENTS = 2**26


def paged_attention_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each head of a request's query attends over the tokens of its context."""
    import torch

    q = decode(row_major_args["query"], point.dtypes["query"])
    keys, values = (
        decode(row_major_args[n], point.dtypes[n]).flatten(0, 1) for n in ("key_cache", "value_cache")
    )
    block_size = point.axes["block_size"]
    table = row_major_args["block_tables"].long()
    positions = torch.arange(table.shape[1] * block_size, device=table.device)
    slots = table.repeat_interleave(block_size, dim=1) * block_size
    slots += positions % block_size
    lengths = row_major_args["context_lens"].long()[:, None]
    visible = positions < lengths
    window = row_major_args["sliding_window"]
    if window > 0:
        visible &= positions >= lengths - window
    group = point.axes["n_heads"] // point.axes["n_kv_heads"]
    out = torch.empty_like(q)
    n_rows = max(1, _CHUNK_ELEMENTS // (positions.numel() * q.shape[1] * q.shape[2]))
    for start in range(0, q.shape[0], n_rows):
        rows = slice(start, start + n_rows)
        k = keys[slots[rows]].repeat_interleave(group, dim=2)
        v = values[slots[rows]].repeat_interleave(group, dim=2)
        logits = torch.einsum("rhd,rthd->rht", q[rows], k) * row_major_args["scale"]
        logits = logits.masked_fill(~visible[rows, None], -torch.inf)
        out[rows] = torch.einsum("rht,rthd->rhd", logits.softmax(dim=-1), v)
    return out


def unified_attention_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each query row attends causally over its request's context, in a window when
    window_size[0] >= 0."""
    import torch

    q = decode(row_major_args["q"], point.dtypes["q"])
    keys, values = (decode(row_major_args[n], point.dtypes[n]).flatten(0, 1) for n in ("k", "v"))
    block_size = point.axes["block_size"]
    n_kv_heads = point.axes["n_kv_heads"]
    group = point.axes["n_heads"] // n_kv_heads
    width = point.attrs["max_seqlen_q"]
    window = point.attrs["window_size"][0]
    table = row_major_args["block_table"].long()
    positions = torch.arange(table.shape[1] * block_size, device=table.device)
    slots = table.repeat_interleave(block_size, dim=1) * block_size
    slots += positions % block_size
    offsets = row_major_args["cu_seqlens_q"].long()
    starts, lengths = offsets[:-1], offsets.diff()
    context = row_major_args["seqused_k"].long()
    # row j of a request sits at context - length + j; rows past its length are padding
    rows = torch.arange(width, device=table.device)
    real = rows < lengths[:, None]
    at = context[:, None] - lengths[:, None] + rows
    visible = positions <= at[..., None]
    if window >= 0:
        visible &= positions >= at[..., None] - window
    index = (starts[:, None] + rows).clamp(max=q.shape[0] - 1)
    out = torch.empty_like(q)
    n_requests = max(1, _CHUNK_ELEMENTS // (positions.numel() * width * q[0].numel()))
    for start in range(0, table.shape[0], n_requests):
        chunk = slice(start, start + n_requests)
        query = q[index[chunk]].unflatten(2, (n_kv_heads, group))
        k, v = keys[slots[chunk]], values[slots[chunk]]
        logits = torch.einsum("rlkgd,rtkd->rlkgt", query, k) * row_major_args["softmax_scale"]
        logits = logits.masked_fill(~visible[chunk, :, None, None], -torch.inf)
        logits = logits.masked_fill(~real[chunk, :, None, None, None], 0.0)
        mixed = torch.einsum("rlkgt,rtkd->rlkgd", logits.softmax(dim=-1), v)
        out[index[chunk][real[chunk]]] = mixed.flatten(2, 3)[real[chunk]]
    return out
