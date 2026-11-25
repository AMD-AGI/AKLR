# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = [
    "FlashMlaWithKvcacheReference",
    "Fp8PagedMqaLogitsReference",
    "Fp8PagedMqaLogitsVerifyReference",
    "MlaDecodeFwdCprrReference",
    "MlaDecodeFwdReference",
]

# Gathered KV values per chunk of rows: 256 MiB of float32.
_CHUNK_ELEMENTS = 2**26


class FlashMlaWithKvcacheReference(OpReference):
    def reference(
        self,
        *,
        q,
        kv_cache,
        kv_cache_format,
        causal,
        sm_scale,
        kv_scale,
        block_table=None,
        cache_seqlens=None,
        cu_seqlens_k=None,
        sparse_indices=None,
        sparse_lens=None,
        extra_kv_cache=None,
        extra_sparse_indices=None,
        extra_sparse_lens=None,
        sinks=None,
        **_,
    ):
        """Each head of a query attends over the KV rows its row reads."""
        dtype, q = q.dtype, q.float()
        batch, query_length = q.shape[:2]
        if kv_cache_format == "dsv4_fp8":
            pools = []
            for cache, indices, lengths in (
                (kv_cache, sparse_indices, sparse_lens),
                (extra_kv_cache, extra_sparse_indices, extra_sparse_lens),
            ):
                if cache is None:
                    continue
                slots = indices[:, 0].long()
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

            def keys_of(rows):
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
                    keys.append(torch.cat((nope, rope), dim=-1).to(q.dtype))
                return torch.cat(keys, dim=1)

            empty_lse = torch.inf
        else:
            pool = kv_cache.flatten(0, 2)
            table = block_table.long()
            if cache_seqlens is not None:
                # a length past the table width reads the whole table row
                lengths = cache_seqlens.long().clamp(max=table.shape[1])
            else:
                offsets = cu_seqlens_k.long()
                lengths = offsets.diff()
                columns = torch.arange(int(lengths.max()), device=table.device)
                table = table[(offsets[:-1, None] + columns).clamp(max=table.numel() - 1)]
            positions = torch.arange(int(lengths.max()), device=table.device)
            # entries past a row's length are padding: they read row 0, hidden by ``visible``
            table = table[:, : positions.numel()].where(positions < lengths[:, None], 0)
            ends = lengths[:, None]
            if causal:
                ends = ends - query_length + 1 + torch.arange(query_length, device=q.device)
            visible = positions < ends[..., None]

            def keys_of(rows):
                """The 576 values of each pool row the rows read, decoded."""
                return pool[table[rows]].float()

            empty_lse = -torch.inf
        sinks = None if sinks is None else sinks.float()
        scale = sm_scale * kv_scale
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
            output[rows] = weights @ (keys[:, None, :, :512] * kv_scale)
            lse[rows] = row_lse.masked_fill(empty, empty_lse)
        return {"output": output.to(dtype), "lse": lse}


class Fp8PagedMqaLogitsReference(OpReference):
    def reference(self, *, q, kv_cache, weights, context_lens, block_tables, **_):
        """Each request's weighted relu logits over the keys of its pages."""
        logits = _paged_mqa_logits(
            q[:, 0].float(), kv_cache, block_tables.long(), weights.float(), context_lens.long()
        )
        return {"logits": logits}


class Fp8PagedMqaLogitsVerifyReference(OpReference):
    def reference(self, *, q, kv_cache, weights, context_lens, block_tables, **_):
        """fp8_paged_mqa_logits of each query row, its request's block table row repeated to it."""
        q = q.float()
        batch, next_n = q.shape[:2]
        logits = _paged_mqa_logits(
            q.flatten(0, 1),
            kv_cache,
            block_tables.long().repeat_interleave(next_n, dim=0),
            weights.float().flatten(0, 1),
            context_lens.long().flatten(),
        )
        return {"logits": logits.view(batch, next_n, -1)}


def _paged_mqa_logits(q, pool, table, weights, context):
    """Each row's weighted relu logits over the keys of its table row's pages, -inf past its context."""
    n_pages, page_size = pool.shape[:2]
    raw = pool.reshape(n_pages, -1)
    # a page's 16 x 16 tiles back to (token, dim) order
    tiles = raw[:, : page_size * 128].reshape(n_pages, page_size // 16, 8, 16, 16)
    keys = tiles.transpose(2, 3).reshape(n_pages, page_size, 128)
    scales = raw[:, page_size * 128 :].contiguous().view(torch.float32)
    keys = keys.view(torch.float8_e4m3fn).to(q.dtype) * scales.to(q.dtype)[..., None]
    out = q.new_empty(table.shape[0], table.shape[1] * page_size)
    # a chunk's gathered keys and its per-head logits: each at most _CHUNK_ELEMENTS
    n_rows = max(1, _CHUNK_ELEMENTS // (out.shape[1] * max(q.shape[1], 128)))
    for start in range(0, out.shape[0], n_rows):
        rows = slice(start, start + n_rows)
        row_keys = keys[table[rows]].flatten(1, 2)  # (rows, max_seq_len, 128)
        logits = torch.einsum("bhd,btd->bht", q[rows], row_keys).relu()
        out[rows] = torch.einsum("bht,bh->bt", logits, weights[rows])
    positions = torch.arange(out.shape[1], device=out.device)
    return out.masked_fill(positions >= context[:, None], -torch.inf)


class MlaDecodeFwdReference(OpReference):
    def reference(self, *, q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale, **_):
        """Each head of a request's query attends over the pool rows its kv_indices name."""
        q = q.float() * q_scale.float().reshape(())
        kv_scale = kv_scale.float().reshape(())
        pool = kv_buffer.flatten(0, 2)
        offsets = kv_indptr.long()
        lengths = offsets.diff()
        positions = torch.arange(max(1, int(lengths.max())), device=q.device)
        visible = positions < lengths[:, None]
        # entries past a request's length read entry 0, hidden by ``visible``
        rows = kv_indices.long()[(offsets[:-1, None] + positions).clamp(max=kv_indices.numel() - 1)]
        rows = rows.where(visible, 0)
        o = q.new_zeros(*q.shape[:-1], 512)
        lse = q.new_full(q.shape[:-1], -torch.inf)
        n_requests = max(1, _CHUNK_ELEMENTS // (positions.numel() * q.shape[-1]))
        for start in range(0, q.shape[0], n_requests):
            requests = slice(start, start + n_requests)
            keys = pool[rows[requests]].float() * kv_scale
            seen = visible[requests][:, None]
            logits = torch.einsum("rhd,rkd->rhk", q[requests], keys) * sm_scale
            logits = logits.masked_fill(~seen, -torch.inf)
            empty = ~seen.any(dim=-1)
            # normalized directly: subtracting lse would carry its rounding at large logits
            weights = logits.softmax(dim=-1).masked_fill(empty[..., None], 0)
            o[requests] = weights @ keys[..., :512]
            lse[requests] = logits.logsumexp(dim=-1).masked_fill(empty, -torch.inf)
        return {"o": o.to(torch.bfloat16), "lse": lse}


class MlaDecodeFwdCprrReference(OpReference):
    def reference(
        self,
        *,
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        g_kv_indptr,
        q_scale,
        kv_scale,
        sm_scale,
        cp_rank,
        cp_world_size,
        **_,
    ):
        """Query i of a request of G global tokens attends over the rows of its shard here whose
        global position, local row j * cp_world_size + cp_rank, is at most G - query_length + i."""
        q = q.float() * q_scale.float().reshape(())
        kv_scale = kv_scale.float().reshape(())
        query_length = q.shape[1]
        pool = kv_buffer.flatten(0, 2)
        offsets = kv_indptr.long()
        lengths = offsets.diff()
        positions = torch.arange(max(1, int(lengths.max())), device=q.device)
        # entries past a shard's length read entry 0, hidden by ``visible``
        rows = kv_indices.long()[(offsets[:-1, None] + positions).clamp(max=kv_indices.numel() - 1)]
        rows = rows.where(positions < lengths[:, None], 0)
        # (request, query, local row)
        query_positions = (
            g_kv_indptr.long().diff()[:, None] - query_length + torch.arange(query_length, device=q.device)
        )
        visible = (positions < lengths[:, None, None]) & (
            positions * cp_world_size + cp_rank <= query_positions[..., None]
        )
        o = q.new_zeros(*q.shape[:-1], 512)
        lse = q.new_full(q.shape[:-1], -torch.inf)
        n_requests = max(1, _CHUNK_ELEMENTS // (positions.numel() * q.shape[-1]))
        for start in range(0, q.shape[0], n_requests):
            requests = slice(start, start + n_requests)
            keys = pool[rows[requests]].float() * kv_scale
            seen = visible[requests][:, :, None]
            logits = torch.einsum("rqhd,rkd->rqhk", q[requests], keys) * sm_scale
            logits = logits.masked_fill(~seen, -torch.inf)
            empty = ~seen.any(dim=-1)
            # normalized directly: subtracting lse would carry its rounding at large logits
            weights = logits.softmax(dim=-1).masked_fill(empty[..., None], 0)
            o[requests] = torch.einsum("rqhk,rkd->rqhd", weights, keys[..., :512])
            lse[requests] = logits.logsumexp(dim=-1).masked_fill(empty, -torch.inf)
        return {"o": o.to(torch.bfloat16), "lse": lse}
