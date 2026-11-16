###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Float64 references of the paged attention ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["paged_attention_decode_reference", "unified_attention_reference"]


def paged_attention_decode_reference(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    scale,
    sliding_window=0,
) -> Tensor:
    """Each head of a request's query attends over the last sliding_window (0: all) tokens of its context."""
    import torch

    q = query.double()
    keys, values = (c.double().flatten(0, 1) for c in (key_cache, value_cache))
    block_size = key_cache.shape[1]
    group = query.shape[1] // key_cache.shape[2]
    table = block_tables.long()
    positions = torch.arange(table.shape[1] * block_size, device=table.device)
    slots = table.repeat_interleave(block_size, dim=1) * block_size + positions % block_size
    lengths = context_lens.long()[:, None]
    visible = positions < lengths
    if sliding_window > 0:
        visible &= positions >= lengths - sliding_window
    out = q.new_empty(q.shape)
    # requests a chunk: 2**26 gathered float64 elements, 512 MiB
    n_rows = max(1, 2**26 // (positions.numel() * q.shape[1] * q.shape[2]))
    for start in range(0, q.shape[0], n_rows):
        chunk = slice(start, start + n_rows)
        k = keys[slots[chunk]].repeat_interleave(group, dim=2)
        v = values[slots[chunk]].repeat_interleave(group, dim=2)
        logits = torch.einsum("rhd,rthd->rht", q[chunk], k) * scale
        logits = logits.masked_fill(~visible[chunk, None], -torch.inf)
        out[chunk] = torch.einsum("rht,rthd->rhd", logits.softmax(dim=-1), v)
    return out


def unified_attention_reference(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    max_seqlen_q,
    window_size=(-1, -1),
) -> Tensor:
    """Each query row attends causally over its request's context, to at most window_size[0] tokens back
    when that is >= 0."""
    import torch

    query = q.double()
    keys, values = (c.double().flatten(0, 1) for c in (k, v))
    block_size, n_kv_heads = k.shape[1], k.shape[2]
    group = q.shape[1] // n_kv_heads
    table = block_table.long()
    positions = torch.arange(table.shape[1] * block_size, device=table.device)
    slots = table.repeat_interleave(block_size, dim=1) * block_size + positions % block_size
    offsets = cu_seqlens_q.long()
    starts, lengths = offsets[:-1], offsets.diff()
    # row j of a request sits at seqused_k - length + j; rows past its length are padding
    rows = torch.arange(max_seqlen_q, device=table.device)
    real = rows < lengths[:, None]
    at = seqused_k.long()[:, None] - lengths[:, None] + rows
    visible = positions <= at[..., None]
    if window_size[0] >= 0:
        visible &= positions >= at[..., None] - window_size[0]
    index = (starts[:, None] + rows).clamp(max=q.shape[0] - 1)
    out = query.new_empty(query.shape)
    # requests a chunk: 2**26 gathered float64 elements, 512 MiB
    n_requests = max(1, 2**26 // (positions.numel() * max_seqlen_q * query[0].numel()))
    for start in range(0, table.shape[0], n_requests):
        chunk = slice(start, start + n_requests)
        grouped = query[index[chunk]].unflatten(2, (n_kv_heads, group))
        logits = torch.einsum("rlkgd,rtkd->rlkgt", grouped, keys[slots[chunk]]) * softmax_scale
        logits = logits.masked_fill(~visible[chunk, :, None, None], -torch.inf)
        logits = logits.masked_fill(~real[chunk, :, None, None, None], 0.0)
        mixed = torch.einsum("rlkgt,rtkd->rlkgd", logits.softmax(dim=-1), values[slots[chunk]])
        out[index[chunk][real[chunk]]] = mixed.flatten(2, 3)[real[chunk]]
    return out
