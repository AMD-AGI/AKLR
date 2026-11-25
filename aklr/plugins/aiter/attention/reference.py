# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["PagedAttentionDecodeReference", "UnifiedAttentionReference"]

# Gathered KV elements per chunk of requests: 256 MiB in float32.
_CHUNK_ELEMENTS = 2**26


class PagedAttentionDecodeReference(OpReference):
    def reference(
        self, *, query, key_cache, value_cache, block_tables, context_lens, scale, sliding_window, **_
    ):
        """Each head of a request's query attends over the tokens of its context."""
        q = query.float()
        block_size, n_kv_heads = key_cache.shape[1], key_cache.shape[2]
        keys, values = key_cache.float().flatten(0, 1), value_cache.float().flatten(0, 1)
        table = block_tables.long()
        positions = torch.arange(table.shape[1] * block_size, device=table.device)
        slots = table.repeat_interleave(block_size, dim=1) * block_size
        slots += positions % block_size
        lengths = context_lens.long()[:, None]
        visible = positions < lengths
        if sliding_window > 0:
            visible &= positions >= lengths - sliding_window
        group = q.shape[1] // n_kv_heads
        out = torch.empty_like(q)
        n_rows = max(1, _CHUNK_ELEMENTS // (positions.numel() * q.shape[1] * q.shape[2]))
        for start in range(0, q.shape[0], n_rows):
            rows = slice(start, start + n_rows)
            k = keys[slots[rows]].repeat_interleave(group, dim=2)
            v = values[slots[rows]].repeat_interleave(group, dim=2)
            logits = torch.einsum("rhd,rthd->rht", q[rows], k) * scale
            logits = logits.masked_fill(~visible[rows, None], -torch.inf)
            out[rows] = torch.einsum("rht,rthd->rhd", logits.softmax(dim=-1), v)
        return {"out": out.to(query.dtype)}


class UnifiedAttentionReference(OpReference):
    def reference(
        self, *, q, k, v, cu_seqlens_q, seqused_k, block_table, softmax_scale, max_seqlen_q, window_size, **_
    ):
        """Each query row attends causally over its request's context, in a window when
        window_size[0] >= 0."""
        dtype, q = q.dtype, q.float()
        block_size, n_kv_heads = k.shape[1], k.shape[2]
        keys, values = k.float().flatten(0, 1), v.float().flatten(0, 1)
        group = q.shape[1] // n_kv_heads
        width = max_seqlen_q
        window = window_size[0]
        table = block_table.long()
        positions = torch.arange(table.shape[1] * block_size, device=table.device)
        slots = table.repeat_interleave(block_size, dim=1) * block_size
        slots += positions % block_size
        offsets = cu_seqlens_q.long()
        starts, lengths = offsets[:-1], offsets.diff()
        context = seqused_k.long()
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
            chunk_keys, chunk_values = keys[slots[chunk]], values[slots[chunk]]
            logits = torch.einsum("rlkgd,rtkd->rlkgt", query, chunk_keys) * softmax_scale
            logits = logits.masked_fill(~visible[chunk, :, None, None], -torch.inf)
            logits = logits.masked_fill(~real[chunk, :, None, None, None], 0.0)
            mixed = torch.einsum("rlkgt,rtkd->rlkgd", logits.softmax(dim=-1), chunk_values)
            out[index[chunk][real[chunk]]] = mixed.flatten(2, 3)[real[chunk]]
        return {"out": out.to(dtype)}
