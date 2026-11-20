###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "flash_mla_with_kvcache_dense_reference",
    "flash_mla_with_kvcache_ragged_reference",
    "flash_mla_with_kvcache_sparse_reference",
    "fp8_paged_mqa_logits_reference",
    "fp8_paged_mqa_logits_verify_reference",
    "mla_decode_fwd_cprr_reference",
    "mla_decode_fwd_reference",
]


def _attend(
    q: Tensor,
    keys_of: Callable[[slice], Tensor],
    visible: Tensor,
    sinks: Tensor | None,
    sm_scale: float | None,
    kv_scale: float,
    empty_lse: float,
) -> tuple[Tensor, Tensor]:
    """Each head of a query attends over the KV rows ``visible`` marks, chunked by
    rows; ``keys_of(rows)`` gathers their keys in float64."""
    import torch

    batch = q.shape[0]
    sinks = None if sinks is None else sinks.double()
    scale = (q.shape[-1] ** -0.5 if sm_scale is None else sm_scale) * kv_scale
    output = q.new_zeros(*q.shape[:-1], 512)
    lse = q.new_full(q.shape[:-1], empty_lse)
    # gathered KV values per chunk of rows: 512 MiB of float64
    n_rows = max(1, 2**26 // (visible.shape[-1] * q.shape[-1]))
    for start in range(0, batch, n_rows):
        rows = slice(start, start + n_rows)
        keys = keys_of(rows)
        seen = visible[rows][:, :, None]
        logits = torch.einsum("rqhd,rkd->rqhk", q[rows], keys) * scale
        logits = logits.masked_fill(~seen, -torch.inf)
        row_lse = logits.logsumexp(dim=-1)
        empty = ~seen.any(dim=-1)
        if sinks is not None:
            logits = torch.cat((logits, sinks.expand_as(row_lse)[..., None]), dim=-1)
        # Normalized directly: subtracting lse would carry its rounding at large logits.
        weights = logits.softmax(dim=-1)[..., : keys.shape[1]]
        weights = weights.masked_fill(empty[..., None], 0)
        output[rows] = weights @ (keys[:, None, :, :512] * kv_scale)
        lse[rows] = row_lse.masked_fill(empty, empty_lse)
    return output, lse


def flash_mla_with_kvcache_sparse_reference(
    q,
    kv_cache,
    sparse_indices=None,
    *,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_k=None,
    causal=False,
    head_dim_v=None,
    sm_scale=None,
    sinks=None,
    kv_cache_format="dsv4_fp8",
    kv_scale=1.0,
) -> tuple[Tensor, Tensor]:
    """Each token attends to the slots its indices name, in the pool and the optional
    extra one, DeepSeek-V4's packed fp8 slots."""
    import torch

    pools = []
    for cache, slots, lengths in (
        (kv_cache, sparse_indices, sparse_lens),
        (extra_kv_cache, extra_sparse_indices, extra_sparse_lens),
    ):
        if cache is None:
            continue
        slots = slots[:, 0].long()
        width = slots.shape[-1]
        selected = slots >= 0
        if lengths is not None:
            if ((lengths < 0) | (lengths > width)).any():
                raise ValueError("a sparse length is outside [0, its index width]")
            positions = torch.arange(width, device=slots.device)
            selected &= positions < lengths[:, None]
        if (slots[selected] >= cache.shape[0] * cache.shape[1]).any():
            raise ValueError("a selected sparse index is outside its KV pool")
        # an unselected slot reads slot 0, hidden by ``visible``
        pools.append((cache, slots.where(selected, 0), selected))
    visible = torch.cat([selected for *_, selected in pools], dim=-1)[:, None]

    def keys_of(rows: slice) -> Tensor:
        # The kernel's literal bf16 bit assembly, not fp8 scaling.
        keys = []
        for cache, slots, _ in pools:
            page_size = cache.shape[1]
            raw = cache.view(torch.uint8).flatten()
            within = (slots[rows] % page_size)[..., None]
            page = (slots[rows] // page_size)[..., None] * (page_size * 584)
            payload = page + within * 576
            arange = torch.arange(448, device=raw.device)
            codes = raw[payload + arange].int()
            scales = raw[page + page_size * 576 + within * 8 + arange[:7]].int()
            exponent = ((codes >> 3) & 15) + scales.repeat_interleave(64, dim=-1) - 7
            bits = ((codes & 128) << 8) | (exponent << 7) | ((codes & 7) << 4)
            nope = (bits & 65535).to(torch.int16).view(torch.bfloat16)
            rope = raw[payload + 448 + arange[:128]].view(torch.bfloat16)
            keys.append(torch.cat((nope, rope), dim=-1).double())
        return torch.cat(keys, dim=1)

    return _attend(q.double(), keys_of, visible, sinks, sm_scale, kv_scale, torch.inf)


def _attend_dense(
    q: Tensor,
    kv_cache: Tensor,
    table: Tensor,
    lengths: Tensor,
    causal: bool,
    sinks: Tensor | None,
    sm_scale: float | None,
    kv_scale: float,
) -> tuple[Tensor, Tensor]:
    """Each request attends to its first ``lengths`` table rows of the pool."""
    import torch

    pool = kv_cache.flatten(0, 2)
    query_length = q.shape[1]
    positions = torch.arange(int(lengths.max()), device=table.device)
    # entries past a row's length are padding: they read row 0, hidden by ``visible``
    table = table[:, : positions.numel()].where(positions < lengths[:, None], 0)
    ends = lengths[:, None]
    if causal:
        ends = ends - query_length + 1 + torch.arange(query_length, device=q.device)
    visible = positions < ends[..., None]

    def keys_of(rows: slice) -> Tensor:
        """The 576 values of each pool row the rows read, in float64."""
        return pool[table[rows]].double()

    return _attend(q.double(), keys_of, visible, sinks, sm_scale, kv_scale, -torch.inf)


def flash_mla_with_kvcache_dense_reference(
    q,
    kv_cache,
    sparse_indices=None,
    *,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_k=None,
    causal=False,
    head_dim_v=None,
    sm_scale=None,
    sinks=None,
    kv_cache_format="dsv4_fp8",
    kv_scale=1.0,
) -> tuple[Tensor, Tensor]:
    """Each request attends to its block_table row, read up to cache_seqlens."""
    table = block_table.long()
    # a length past the table width reads the whole table row
    lengths = cache_seqlens.long().clamp(max=table.shape[1])
    return _attend_dense(q, kv_cache, table, lengths, causal, sinks, sm_scale, kv_scale)


def flash_mla_with_kvcache_ragged_reference(
    q,
    kv_cache,
    sparse_indices=None,
    *,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_k=None,
    causal=False,
    head_dim_v=None,
    sm_scale=None,
    sinks=None,
    kv_cache_format="dsv4_fp8",
    kv_scale=1.0,
) -> tuple[Tensor, Tensor]:
    """Each request attends to its rows of the flat block_table, cu_seqlens_k apart."""
    import torch

    table = block_table.long()
    offsets = cu_seqlens_k.long()
    lengths = offsets.diff()
    columns = torch.arange(int(lengths.max()), device=table.device)
    table = table[(offsets[:-1, None] + columns).clamp(max=table.numel() - 1)]
    return _attend_dense(q, kv_cache, table, lengths, causal, sinks, sm_scale, kv_scale)


def fp8_paged_mqa_logits_reference(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    logits,
    *,
    page_size,
    preshuffled=True,
) -> Tensor:
    """Each request's weighted relu logits over the keys of its pages."""
    return _paged_mqa_logits(q[:, 0], kv_cache, weights, context_lens, block_tables, page_size)


def fp8_paged_mqa_logits_verify_reference(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    logits,
    *,
    page_size,
    preshuffled=True,
) -> Tensor:
    """fp8_paged_mqa_logits of each query row, its request's block table row repeated to it."""
    batch, next_n = q.shape[:2]
    rows = _paged_mqa_logits(
        q.flatten(0, 1),
        kv_cache,
        weights.flatten(0, 1),
        context_lens.flatten(),
        block_tables.repeat_interleave(next_n, dim=0),
        page_size,
    )
    return rows.view(batch, next_n, -1)


def _paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, page_size) -> Tensor:
    """Each row's weighted relu logits over the keys of its table row's pages, -inf past its context."""
    import torch

    n_pages = kv_cache.shape[0]
    raw = kv_cache.reshape(n_pages, -1)
    # a page's 16 x 16 tiles back to (token, dim) order
    tiles = raw[:, : page_size * 128].reshape(n_pages, page_size // 16, 8, 16, 16)
    keys = tiles.transpose(2, 3).reshape(n_pages, page_size, 128)
    scales = raw[:, page_size * 128 :].contiguous().view(torch.float32)
    keys = keys.view(torch.float8_e4m3fn).double() * scales.double()[..., None]
    table = block_tables.long()
    queries = q.double()
    out = queries.new_empty(table.shape[0], table.shape[1] * page_size)
    # a chunk's gathered keys and its per-head logits: each at most 2**26 elements
    n_rows = max(1, 2**26 // (out.shape[1] * max(queries.shape[1], 128)))
    for start in range(0, out.shape[0], n_rows):
        rows = slice(start, start + n_rows)
        row_keys = keys[table[rows]].flatten(1, 2)  # (rows, max_seq_len, 128)
        scores = torch.einsum("bhd,btd->bht", queries[rows], row_keys).relu()
        out[rows] = torch.einsum("bht,bh->bt", scores, weights[rows].double())
    positions = torch.arange(out.shape[1], device=out.device)
    return out.masked_fill(positions >= context_lens.long()[:, None], -torch.inf)


def mla_decode_fwd_reference(
    q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale
) -> tuple[Tensor, Tensor]:
    """Each head of a request's query attends over the pool rows its kv_indices name."""
    import torch

    queries = q.float() * q_scale.reshape(())
    kv_scale = kv_scale.reshape(())
    pool = kv_buffer.flatten(0, 2)
    offsets = kv_indptr.long()
    lengths = offsets.diff()
    positions = torch.arange(max(1, int(lengths.max())), device=q.device)
    visible = positions < lengths[:, None]
    # entries past a request's length read entry 0, hidden by visible
    rows = kv_indices.long()[(offsets[:-1, None] + positions).clamp(max=kv_indices.numel() - 1)]
    rows = rows.where(visible, 0)
    o = queries.new_zeros(*queries.shape[:-1], 512)
    lse = queries.new_full(queries.shape[:-1], -torch.inf)
    # a chunk's gathered keys: at most 2**26 float32 elements
    n_requests = max(1, 2**26 // (positions.numel() * queries.shape[-1]))
    for start in range(0, queries.shape[0], n_requests):
        requests = slice(start, start + n_requests)
        keys = pool[rows[requests]].float() * kv_scale
        seen = visible[requests][:, None]
        logits = torch.einsum("rhd,rkd->rhk", queries[requests], keys) * sm_scale
        logits = logits.masked_fill(~seen, -torch.inf)
        empty = ~seen.any(dim=-1)
        # normalized directly: subtracting lse would carry its rounding at large logits
        weights = logits.softmax(dim=-1).masked_fill(empty[..., None], 0)
        o[requests] = weights @ keys[..., :512]
        lse[requests] = logits.logsumexp(dim=-1).masked_fill(empty, -torch.inf)
    return o, lse


def mla_decode_fwd_cprr_reference(
    q, kv_buffer, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, cp_world_size, sm_scale, cp_rank=0
) -> tuple[Tensor, Tensor]:
    """Query i of a request of G global tokens attends over the rows of its shard here whose global
    position, local row j * cp_world_size + cp_rank, is at most G - query_length + i."""
    import torch

    queries = q.float() * q_scale.reshape(())
    kv_scale = kv_scale.reshape(())
    query_length = queries.shape[1]
    pool = kv_buffer.flatten(0, 2)
    offsets = kv_indptr.long()
    lengths = offsets.diff()
    positions = torch.arange(max(1, int(lengths.max())), device=q.device)
    # entries past a shard's length read entry 0, hidden by visible
    rows = kv_indices.long()[(offsets[:-1, None] + positions).clamp(max=kv_indices.numel() - 1)]
    rows = rows.where(positions < lengths[:, None], 0)
    # (request, query, local row)
    query_positions = (
        g_kv_indptr.long().diff()[:, None] - query_length + torch.arange(query_length, device=q.device)
    )
    visible = (positions < lengths[:, None, None]) & (
        positions * cp_world_size + cp_rank <= query_positions[..., None]
    )
    o = queries.new_zeros(*queries.shape[:-1], 512)
    lse = queries.new_full(queries.shape[:-1], -torch.inf)
    # a chunk's gathered keys: at most 2**26 float32 elements
    n_requests = max(1, 2**26 // (positions.numel() * queries.shape[-1]))
    for start in range(0, queries.shape[0], n_requests):
        requests = slice(start, start + n_requests)
        keys = pool[rows[requests]].float() * kv_scale
        seen = visible[requests][:, :, None]
        logits = torch.einsum("rqhd,rkd->rqhk", queries[requests], keys) * sm_scale
        logits = logits.masked_fill(~seen, -torch.inf)
        empty = ~seen.any(dim=-1)
        # normalized directly: subtracting lse would carry its rounding at large logits
        weights = logits.softmax(dim=-1).masked_fill(empty[..., None], 0)
        o[requests] = torch.einsum("rqhk,rkd->rqhd", weights, keys[..., :512])
        lse[requests] = logits.logsumexp(dim=-1).masked_fill(empty, -torch.inf)
    return o, lse
