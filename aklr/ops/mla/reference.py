# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["flash_mla_with_kvcache_reference", "fp8_paged_mqa_logits_reference", "mla_decode_fwd_reference"]

# Gathered KV values per chunk of rows: 256 MiB of float32.
_CHUNK_ELEMENTS = 2**26


def flash_mla_with_kvcache_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """Each head of a query attends over the KV rows its row reads. With has_inv_rope, the
    output's last 64 lanes are then rotated back: per (cos, sin) pair of
    rope_freqs[positions], even * cos + odd * sin and odd * cos - even * sin, on the unrounded
    output (the kernels' bf16 rounding of the output before the rotation is error left to
    compare). Without return_lse, lse is empty."""
    import torch

    q = decode(row_major_args["q"], point.dtypes["q"])
    batch, query_length = q.shape[:2]
    if point.attrs["kv_cache_format"] == "dsv4_fp8":
        pools = []
        for prefix in ("", "extra_"):
            cache = row_major_args.get(prefix + "kv_cache")
            if cache is None:
                continue
            slots = row_major_args[prefix + "sparse_indices"][:, 0].long()
            width = slots.shape[-1]
            selected = slots >= 0
            lengths = row_major_args.get(prefix + "sparse_lens")
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

        # with the inverse RoPE, the slots decode as aiter's kernel (sglang's) decodes them
        exact_fp8 = point.attrs.get("has_inv_rope", False)

        def keys_of(rows: slice) -> Tensor:
            # The kernel's literal bf16 bit assembly, not fp8 scaling (exact_fp8: fp8 scaling).
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
                if exact_fp8:
                    # aiter's scaled upcast: the fp8 value (subnormals and zero as e4m3
                    # defines them) times 2 ** (exponent - 127)
                    fp8 = codes.to(torch.uint8).view(torch.float8_e4m3fn).to(q.dtype)
                    nope = fp8 * torch.exp2(scales.to(q.dtype) - 127).repeat_interleave(64, dim=-1)
                    rope = raw[payload + 448 + arange[:128]].view(torch.bfloat16)
                    keys.append(torch.cat((nope, rope.to(q.dtype)), dim=-1))
                    continue
                exponent = ((codes >> 3) & 15) + scales.repeat_interleave(64, dim=-1) - 7
                bits = ((codes & 128) << 8) | (exponent << 7) | ((codes & 7) << 4)
                nope = (bits & 65535).to(torch.int16).view(torch.bfloat16)
                rope = raw[payload + 448 + arange[:128]].view(torch.bfloat16)
                keys.append(torch.cat((nope, rope), dim=-1).to(q.dtype))
            return torch.cat(keys, dim=1)

        empty_lse = torch.inf
    else:
        pool = row_major_args["kv_cache"].flatten(0, 2)
        table = row_major_args["block_table"].long()
        if point.attrs["has_cache_seqlens"]:
            # a length past the table width reads the whole table row
            lengths = row_major_args["cache_seqlens"].long().clamp(max=table.shape[1])
        else:
            offsets = row_major_args["cu_seqlens_k"].long()
            lengths = offsets.diff()
            columns = torch.arange(int(lengths.max()), device=table.device)
            table = table[(offsets[:-1, None] + columns).clamp(max=table.numel() - 1)]
        positions = torch.arange(int(lengths.max()), device=table.device)
        # entries past a row's length are padding: they read row 0, hidden by ``visible``
        table = table[:, : positions.numel()].where(positions < lengths[:, None], 0)
        ends = lengths[:, None]
        if point.attrs["causal"]:
            ends = ends - query_length + 1 + torch.arange(query_length, device=q.device)
        visible = positions < ends[..., None]

        def keys_of(rows: slice) -> Tensor:
            """The 576 values of each pool row the rows read, decoded."""
            return decode(pool[table[rows]], point.dtypes["kv_cache"])

        empty_lse = -torch.inf
    sinks = row_major_args.get("sinks")
    sinks = None if sinks is None else decode(sinks, point.dtypes["sinks"])
    scale = row_major_args["sm_scale"] * row_major_args["kv_scale"]
    output = q.new_zeros(*q.shape[:-1], 512)
    lse = q.new_full(q.shape[:-1], empty_lse)
    n_rows = max(1, _CHUNK_ELEMENTS // (visible.shape[-1] * q.shape[-1]))
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
        output[rows] = weights @ (keys[:, None, :, :512] * row_major_args["kv_scale"])
        lse[rows] = row_lse.masked_fill(empty, empty_lse)
    if point.attrs.get("has_inv_rope", False):
        freqs = decode(row_major_args["rope_freqs"], point.dtypes["rope_freqs"])
        freqs = freqs[row_major_args["positions"].long()][:, None, None]
        cos, sin = freqs[..., 0::2], freqs[..., 1::2]
        rope_dim = freqs.shape[-1]
        even, odd = output[..., -rope_dim::2], output[..., -rope_dim + 1 :: 2]
        rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
        output = torch.cat((output[..., :-rope_dim], rotated.flatten(-2)), dim=-1)
    if not point.attrs["return_lse"]:
        lse = lse[..., :0]
    return output, lse


def fp8_paged_mqa_logits_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each request's weighted relu logits over the keys of its pages."""
    import torch

    q = decode(row_major_args["q"][:, 0], point.dtypes["q"])
    pool = row_major_args["kv_cache"]
    n_pages, page_size = pool.shape[:2]
    raw = pool.reshape(n_pages, -1)
    # a page's 16 x 16 tiles back to (token, dim) order
    tiles = raw[:, : page_size * 128].reshape(n_pages, page_size // 16, 8, 16, 16)
    keys = tiles.transpose(2, 3).reshape(n_pages, page_size, 128)
    scales = raw[:, page_size * 128 :].contiguous().view(torch.float32)
    keys = keys.view(torch.float8_e4m3fn).to(q.dtype) * scales.to(q.dtype)[..., None]
    table = row_major_args["block_tables"].long()
    weights = decode(row_major_args["weights"], "float32")
    out = q.new_empty(table.shape[0], table.shape[1] * page_size)
    # a chunk's gathered keys and its per-head logits: each at most _CHUNK_ELEMENTS
    n_rows = max(1, _CHUNK_ELEMENTS // (out.shape[1] * max(q.shape[1], 128)))
    for start in range(0, out.shape[0], n_rows):
        rows = slice(start, start + n_rows)
        row_keys = keys[table[rows]].flatten(1, 2)  # (rows, max_seq_len, 128)
        logits = torch.einsum("bhd,btd->bht", q[rows], row_keys).relu()
        out[rows] = torch.einsum("bht,bh->bt", logits, weights[rows])
    context = row_major_args["context_lens"].long()
    positions = torch.arange(out.shape[1], device=out.device)
    return out.masked_fill(positions >= context[:, None], -torch.inf)


def mla_decode_fwd_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """Each head of a request's query attends over the pool rows its kv_indices name."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    q = decode(args["q"], dtypes["q"]).float() * decode(args["q_scale"], dtypes["q_scale"]).reshape(())
    kv_scale = decode(args["kv_scale"], dtypes["kv_scale"]).reshape(())
    pool = args["kv_buffer"].flatten(0, 2)
    offsets = args["kv_indptr"].long()
    lengths = offsets.diff()
    positions = torch.arange(max(1, int(lengths.max())), device=q.device)
    visible = positions < lengths[:, None]
    # entries past a request's length read entry 0, hidden by ``visible``
    rows = args["kv_indices"].long()[
        (offsets[:-1, None] + positions).clamp(max=args["kv_indices"].numel() - 1)
    ]
    rows = rows.where(visible, 0)
    o = q.new_zeros(*q.shape[:-1], 512)
    lse = q.new_full(q.shape[:-1], -torch.inf)
    n_requests = max(1, _CHUNK_ELEMENTS // (positions.numel() * q.shape[-1]))
    for start in range(0, q.shape[0], n_requests):
        requests = slice(start, start + n_requests)
        keys = decode(pool[rows[requests]], dtypes["kv_buffer"]).float() * kv_scale
        seen = visible[requests][:, None]
        logits = torch.einsum("rhd,rkd->rhk", q[requests], keys) * args["sm_scale"]
        logits = logits.masked_fill(~seen, -torch.inf)
        empty = ~seen.any(dim=-1)
        # normalized directly: subtracting lse would carry its rounding at large logits
        weights = logits.softmax(dim=-1).masked_fill(empty[..., None], 0)
        o[requests] = weights @ keys[..., :512]
        lse[requests] = logits.logsumexp(dim=-1).masked_fill(empty, -torch.inf)
    return o, lse
