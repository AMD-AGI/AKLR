# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels for sglang's per-step DSV4 attention metadata, one launch each, writing the
same integers as sglang's torch ops; the ``*_disagrees`` helpers compare them with sglang's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_kernel(
    seq_ptr,
    req_ptr,
    seq_out,
    req_out,
    n,
    seq_stride,
    req_stride,
    QO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    b = i // QO
    s = tl.load(seq_ptr + b * seq_stride, mask=live, other=0)
    r = tl.load(req_ptr + b * req_stride, mask=live, other=0)
    tl.store(seq_out + i, s + (i % QO - QO + 1), mask=live)
    tl.store(req_out + i, r, mask=live)


def expand(bs: int, qo_len: int, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor):
    """``expand_extend_with_same_length``: ``(seq_lens_casual, req_pool_indices_repeated)``,
    ``[bs * qo_len]`` each, in sglang's dtypes."""
    n = bs * qo_len
    seq_dtype = torch.promote_types(seq_lens.dtype, torch.int32)
    seq_out = torch.empty(n, dtype=seq_dtype, device=seq_lens.device)
    req_out = torch.empty(n, dtype=req_pool_indices.dtype, device=req_pool_indices.device)
    if n:
        block = 64 if n <= 64 else 256
        _expand_kernel[(triton.cdiv(n, block),)](
            seq_lens,
            req_pool_indices,
            seq_out,
            req_out,
            n,
            seq_lens.stride(0),
            req_pool_indices.stride(0),
            QO=qo_len,
            BLOCK=block,
        )
    return seq_out, req_out


@triton.jit
def _swa_kernel(
    seq_ptr,
    req_ptr,
    r2t_ptr,
    map_ptr,
    out_ptr,
    r2t_stride,
    map_n,
    W: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, W)
    pos = tl.load(seq_ptr + row).to(tl.int32) - 1
    off = pos - k
    before = off < 0
    req = tl.load(req_ptr + row).to(tl.int64)
    raw = tl.load(r2t_ptr + req * r2t_stride + tl.where(before, 0, off)).to(tl.int64)
    raw = tl.where(before, -1, raw)
    loc = tl.load(map_ptr + tl.where(raw < 0, raw + map_n, raw))
    tl.store(out_ptr + row * W + k, loc.to(tl.int32))


def swa_page_indices(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    req_to_token: torch.Tensor,
    mapping: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """``get_swa_page_indices``: ``[rows, window]`` int32."""
    rows = seq_lens_casual.shape[0]
    out = torch.empty((rows, window), dtype=torch.int32, device=seq_lens_casual.device)
    if rows:
        _swa_kernel[(rows,)](
            seq_lens_casual,
            req_pool_indices_repeated,
            req_to_token,
            mapping,
            out,
            req_to_token.stride(0),
            mapping.shape[0],
            W=window,
        )
    return out


@triton.jit
def _verify_rows_kernel(
    seq_ptr,
    req_ptr,
    ext_ptr,
    r2t_ptr,
    map_ptr,
    seq_out,
    ext_out,
    casual_out,
    req_out,
    pos_out,
    swa_out,
    swa_len_out,
    loc_out,
    zero_out,
    bs,
    rows,
    r2t_stride,
    map_n,
    QO: tl.constexpr,
    W: tl.constexpr,
    HAS_EXT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // QO
    s = tl.load(seq_ptr + b) + QO
    if row < bs:
        sr = tl.load(seq_ptr + row) + QO
        tl.store(seq_out + row, sr)
        if HAS_EXT:
            e = tl.load(ext_ptr + row).to(sr.dtype)
        else:
            e = sr * 0 + QO
        tl.store(ext_out + row, tl.maximum(tl.minimum(e, sr), 1))
    casual = (s + (row % QO - QO + 1)).to(tl.int32)
    req = tl.load(req_ptr + b)
    tl.store(casual_out + row, casual)
    tl.store(req_out + row, req)
    tl.store(pos_out + row, casual - 1)
    tl.store(swa_len_out + row, tl.minimum(casual, W))
    tl.store(zero_out + row, 0)
    tl.store(loc_out + row, -1)
    tl.store(loc_out + rows + row, -1)
    k = tl.arange(0, W)
    off = casual - 1 - k
    before = off < 0
    raw = tl.load(r2t_ptr + req.to(tl.int64) * r2t_stride + tl.where(before, 0, off)).to(tl.int64)
    raw = tl.where(before, -1, raw)
    loc = tl.load(map_ptr + tl.where(raw < 0, raw + map_n, raw))
    tl.store(swa_out + row * W + k, loc.to(tl.int32))


def verify_rows(
    *,
    seq_lens,
    req_pool_indices,
    extend_seq_lens,
    qo_len,
    req_to_token,
    mapping,
    window,
):
    """make_forward_metadata_from_raw_verify's row pieces in one launch, in return order; the
    ratio-4 / ratio-128 placeholders are out_loc [2, rows] of -1 and lengths [rows] of 0.
    """
    bs = seq_lens.shape[0]
    rows = bs * qo_len
    dev = seq_lens.device
    i32 = {"dtype": torch.int32, "device": dev}
    seq_out = torch.empty_like(seq_lens)
    ext_out = torch.empty_like(seq_lens)
    casual = torch.empty(rows, **i32)
    req_out = torch.empty(rows, dtype=req_pool_indices.dtype, device=dev)
    pos = torch.empty(rows, **i32)
    swa = torch.empty((rows, window), **i32)
    swa_len = torch.empty(rows, **i32)
    loc = torch.empty((2, rows), dtype=torch.int64, device=dev)
    zero = torch.empty(rows, **i32)
    if rows:
        has_ext = extend_seq_lens is not None
        _verify_rows_kernel[(rows,)](
            seq_lens,
            req_pool_indices,
            extend_seq_lens if has_ext else seq_lens,
            req_to_token,
            mapping,
            seq_out,
            ext_out,
            casual,
            req_out,
            pos,
            swa,
            swa_len,
            loc,
            zero,
            bs,
            rows,
            req_to_token.stride(0),
            mapping.shape[0],
            QO=qo_len,
            W=window,
            HAS_EXT=has_ext,
        )
    return seq_out, ext_out, casual, req_out, pos, swa, swa_len, loc, zero


@triton.jit
def _verify_meta_kernel(
    seq_ptr,
    req_ptr,
    ext_ptr,
    r2t_ptr,
    map_ptr,
    loc_ptr,
    count_ptr,
    seq_out,
    ext_out,
    casual_out,
    req_out,
    pos_out,
    swa_out,
    swa_len_out,
    c4c128_loc_out,
    zero_out,
    sparse_zero_out,
    c1_loc_out,
    c1_len_out,
    c2_loc_out,
    c2_len_out,
    c1_sparse_len_out,
    c2_sparse_len_out,
    neg_out,
    pt_out,
    pt1_out,
    pt2_out,
    bs,
    rows,
    r2t_stride,
    map_n,
    num_cols,
    topk,
    QO: tl.constexpr,
    W: tl.constexpr,
    WIDTH: tl.constexpr,
    NEG: tl.constexpr,
    PAGE: tl.constexpr,
    BPP1: tl.constexpr,
    BPP2: tl.constexpr,
    HAS_EXT: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // QO
    req = tl.load(req_ptr + b)
    if COUNT:  # a build-time and a run-time test
        if (row == 0) & (blk == 0):
            tl.store(count_ptr, tl.load(count_ptr) + 1)
    # the page table's columns of this block, and both indexer pools' block tables
    col = blk * BLOCK + tl.arange(0, BLOCK)
    cm = col < num_cols
    v = tl.load(
        r2t_ptr + req.to(tl.int64) * r2t_stride + col.to(tl.int64) * PAGE,
        mask=cm,
        other=0,
    )
    q = v // PAGE
    q = tl.where((v < 0) & (v % PAGE != 0), q - 1, q)
    tl.store(pt_out + row.to(tl.int64) * num_cols + col, q.to(tl.int32), mask=cm)
    base = q.to(tl.int32).to(tl.int64)
    k1 = tl.arange(0, BPP1)
    tl.store(
        pt1_out + row.to(tl.int64) * (num_cols * BPP1) + col[:, None] * BPP1 + k1[None, :],
        (base[:, None] * BPP1 + k1[None, :]).to(tl.int32),
        mask=cm[:, None],
    )
    k2 = tl.arange(0, BPP2)
    tl.store(
        pt2_out + row.to(tl.int64) * (num_cols * BPP2) + col[:, None] * BPP2 + k2[None, :],
        (base[:, None] * BPP2 + k2[None, :]).to(tl.int32),
        mask=cm[:, None],
    )
    if blk == 0:
        s = tl.load(seq_ptr + b) + QO
        if row < bs:
            sr = tl.load(seq_ptr + row) + QO
            tl.store(seq_out + row, sr)
            if HAS_EXT:
                e = tl.load(ext_ptr + row).to(sr.dtype)
            else:
                e = sr * 0 + QO
            tl.store(ext_out + row, tl.maximum(tl.minimum(e, sr), 1))
        casual = (s + (row % QO - QO + 1)).to(tl.int32)
        tl.store(casual_out + row, casual)
        tl.store(req_out + row, req)
        tl.store(pos_out + row, casual - 1)
        tl.store(swa_len_out + row, tl.minimum(casual, W))
        tl.store(zero_out + row, 0)
        tl.store(sparse_zero_out + row, 0)
        tl.store(sparse_zero_out + rows + row, 0)
        tl.store(c4c128_loc_out + row, -1)
        tl.store(c4c128_loc_out + rows + row, -1)
        loc = tl.load(loc_ptr + row).to(tl.int64)
        c1_len = tl.maximum(casual, 1)
        c2_len = tl.maximum(casual // 2, 1)
        tl.store(c1_loc_out + row, loc)
        tl.store(c1_len_out + row, c1_len)
        tl.store(c2_loc_out + row, tl.where((casual % 2) == 0, loc >> 1, -1))
        tl.store(c2_len_out + row, c2_len)
        tl.store(c1_sparse_len_out + row, tl.minimum(c1_len, topk))
        tl.store(c2_sparse_len_out + row, tl.minimum(c2_len, topk))
        # the -1 top-k buffers (ratio 4, ratio 1 and 2 page and raw indices), a row each
        w = tl.arange(0, WIDTH)
        for j in tl.static_range(NEG):
            tl.store(neg_out + (j * rows + row).to(tl.int64) * WIDTH + w, -1)
        k = tl.arange(0, W)
        off = casual - 1 - k
        before = off < 0
        raw = tl.load(r2t_ptr + req.to(tl.int64) * r2t_stride + tl.where(before, 0, off)).to(tl.int64)
        raw = tl.where(before, -1, raw)
        sw = tl.load(map_ptr + tl.where(raw < 0, raw + map_n, raw))
        tl.store(swa_out + row * W + k, sw.to(tl.int32))


def verify_meta(
    *,
    seq_lens,
    req_pool_indices,
    extend_seq_lens,
    qo_len,
    req_to_token,
    mapping,
    window,
    raw_out_loc,
    max_seq_len,
    page_size,
    index_topk,
    width,
    bpp1,
    bpp2,
    counter=None,
):
    """make_forward_metadata_from_raw_verify's tensors of a served step in one launch, a dict
    by sglang's attribute names; the indexers' block tables have bpp1 / bpp2 blocks a page.
    With ``counter`` (int64 [1]) one is added to it."""
    bs = seq_lens.shape[0]
    rows = bs * qo_len
    dev = seq_lens.device
    i32 = {"dtype": torch.int32, "device": dev}
    i64 = {"dtype": torch.int64, "device": dev}
    num_cols = max(0, (min(max_seq_len, req_to_token.shape[1]) + page_size - 1) // page_size)
    neg = torch.empty((5, rows, width), **i32)
    out = {
        "seq_lens": torch.empty_like(seq_lens),
        "extend_seq_lens": torch.empty_like(seq_lens),
        "seq_lens_casual": torch.empty(rows, **i32),
        "req_pool_indices_repeated": torch.empty(rows, dtype=req_pool_indices.dtype, device=dev),
        "positions_casual": torch.empty(rows, **i32),
        "swa_page_indices": torch.empty((rows, window), **i32),
        "swa_topk_lengths": torch.empty(rows, **i32),
        "c4c128_out_loc": torch.empty((2, rows), **i64),
        "zero_lengths": torch.empty(rows, **i32),
        "sparse_zero": torch.empty((2, rows), **i32),
        "c1_out_loc": torch.empty(rows, **i64),
        "c1_topk_lengths_clamp1": torch.empty(rows, **i32),
        "c2_out_loc": torch.empty(rows, **i64),
        "c2_topk_lengths_clamp1": torch.empty(rows, **i32),
        "c1_sparse_topk_lengths": torch.empty(rows, **i32),
        "c2_sparse_topk_lengths": torch.empty(rows, **i32),
        "c4_sparse_page_indices": neg[0],
        "c1_sparse_page_indices": neg[1],
        "c2_sparse_page_indices": neg[2],
        "c1_sparse_raw_indices": neg[3],
        "c2_sparse_raw_indices": neg[4],
        "page_table": torch.empty((rows, num_cols), **i32),
        "c1_page_table": torch.empty((rows, num_cols * bpp1), **i32),
        "c2_page_table": torch.empty((rows, num_cols * bpp2), **i32),
    }
    if rows:
        has_ext = extend_seq_lens is not None
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        _verify_meta_kernel[(rows, max(1, triton.cdiv(num_cols, block)))](
            seq_lens,
            req_pool_indices,
            extend_seq_lens if has_ext else seq_lens,
            req_to_token,
            mapping,
            raw_out_loc,
            counter if counter is not None else raw_out_loc,
            out["seq_lens"],
            out["extend_seq_lens"],
            out["seq_lens_casual"],
            out["req_pool_indices_repeated"],
            out["positions_casual"],
            out["swa_page_indices"],
            out["swa_topk_lengths"],
            out["c4c128_out_loc"],
            out["zero_lengths"],
            out["sparse_zero"],
            out["c1_out_loc"],
            out["c1_topk_lengths_clamp1"],
            out["c2_out_loc"],
            out["c2_topk_lengths_clamp1"],
            out["c1_sparse_topk_lengths"],
            out["c2_sparse_topk_lengths"],
            neg,
            out["page_table"],
            out["c1_page_table"],
            out["c2_page_table"],
            bs,
            rows,
            req_to_token.stride(0),
            mapping.shape[0],
            num_cols,
            index_topk,
            QO=qo_len,
            W=window,
            WIDTH=width,
            NEG=5,
            PAGE=page_size,
            BPP1=bpp1,
            BPP2=bpp2,
            HAS_EXT=has_ext,
            COUNT=counter is not None,
            BLOCK=block,
            num_warps=4,
        )
    return out


@triton.jit
def _swa_loc_kernel(loc_ptr, map_ptr, out32, out64, n, map_n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    loc = tl.load(loc_ptr + i, mask=live, other=0).to(tl.int64)
    v = tl.load(map_ptr + tl.where(loc < 0, loc + map_n, loc), mask=live, other=0)
    tl.store(out32 + i, v.to(tl.int32), mask=live)
    tl.store(out64 + i, v.to(tl.int32).to(tl.int64), mask=live)


def swa_loc(out_cache_loc, mapping):
    """``translate_loc_from_full_to_swa(out_cache_loc).to(int32)`` in one launch, and the same
    widened to int64 (K1's slots)."""
    n = out_cache_loc.shape[0]
    out32 = torch.empty(n, dtype=torch.int32, device=out_cache_loc.device)
    out64 = torch.empty(n, dtype=torch.int64, device=out_cache_loc.device)
    if n:
        block = 64 if n <= 64 else 256
        _swa_loc_kernel[(triton.cdiv(n, block),)](
            out_cache_loc, mapping, out32, out64, n, mapping.shape[0], BLOCK=block
        )
    return out32, out64


@triton.jit
def _block_rows_kernel(
    seq_ptr,
    req_ptr,
    casual_out,
    req_out,
    pos_out,
    n,
    QO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    b = i // QO
    s = tl.load(seq_ptr + b, mask=live, other=0) + QO
    casual = (s + (i % QO - QO + 1)).to(tl.int32)
    tl.store(casual_out + i, casual, mask=live)
    tl.store(req_out + i, tl.load(req_ptr + b, mask=live, other=0), mask=live)
    tl.store(pos_out + i, casual - 1, mask=live)


def block_rows(seq_lens, req_pool_indices, qo_len: int):
    """init_forward_metadata_dspark_draft_block's rows in one launch: (seq_lens_casual int32 of
    ``seq_lens + qo_len``, req_pool_indices_repeated, positions int32)."""
    n = seq_lens.shape[0] * qo_len
    dev = seq_lens.device
    casual = torch.empty(n, dtype=torch.int32, device=dev)
    req_out = torch.empty(n, dtype=req_pool_indices.dtype, device=dev)
    pos = torch.empty(n, dtype=torch.int32, device=dev)
    if n:
        block = 64 if n <= 64 else 256
        _block_rows_kernel[(triton.cdiv(n, block),)](
            seq_lens, req_pool_indices, casual, req_out, pos, n, QO=qo_len, BLOCK=block
        )
    return casual, req_out, pos


# scratch fields whose bytes are left unwritten (sglang's small-batch top-k plan): dtype and
# shape only
_UNWRITTEN = ("topk_metadata", "topk_metadata_chunks")


@triton.jit
def _block_meta_kernel(
    seq_ptr,
    req_ptr,
    r2t_ptr,
    map_ptr,
    loc_ptr,
    valid_ptr,
    count_ptr,
    casual_out,
    req_out,
    pos_out,
    swa_out,
    topk_out,
    pt_out,
    r2t_stride,
    num_cols,
    QO: tl.constexpr,
    W: tl.constexpr,
    TW: tl.constexpr,
    TW_BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    blk = tl.program_id(1)
    i = q // QO
    req = tl.load(req_ptr + i)
    col = blk * BLOCK + tl.arange(0, BLOCK)
    cm = col < num_cols
    v = tl.load(
        r2t_ptr + req.to(tl.int64) * r2t_stride + col.to(tl.int64) * PAGE,
        mask=cm,
        other=0,
    )
    pq = v // PAGE
    pq = tl.where((v < 0) & (v % PAGE != 0), pq - 1, pq)
    tl.store(pt_out + q.to(tl.int64) * num_cols + col, pq.to(tl.int32), mask=cm)
    if blk == 0:
        if COUNT:  # a build-time and a run-time test
            if q == 0:
                tl.store(count_ptr, tl.load(count_ptr) + 1)
        seq = tl.load(seq_ptr + i)
        casual = (seq + QO + (q % QO - QO + 1)).to(tl.int32)
        tl.store(casual_out + q, casual)
        tl.store(req_out + q, req)
        tl.store(pos_out + q, casual - 1)
        # the block's window (ComputeDsparkWindowGather) and SWA page indices (BuildDsparkSwaPageIndices)
        prefix = (seq + 1).to(tl.int32).to(tl.int64) - 1
        cl = tl.minimum(prefix, W)
        if HAS_VALID:
            valid_from = tl.load(valid_ptr + req.to(tl.int64))
            cl = tl.minimum(cl, tl.maximum(prefix - valid_from, 0))
        cl = cl.to(tl.int32)
        rp = req.to(tl.int64)
        k = tl.arange(0, TW_BLOCK)
        kmask = k < TW
        in_window = k < cl
        src_col = tl.minimum(tl.maximum((W - cl) + k, 0), W - 1)
        wmask = kmask & in_window
        off = tl.maximum(prefix - W + src_col, 0)
        win_full = tl.load(r2t_ptr + rp * r2t_stride + off, mask=wmask, other=0).to(tl.int64)
        win_swa = tl.load(map_ptr + win_full, mask=wmask, other=-1).to(tl.int32)
        in_block = (k >= cl) & (k < cl + QO)
        bmask = kmask & in_block
        bcol = tl.maximum(k - cl, 0)
        blk_full = tl.load(loc_ptr + i * QO + bcol, mask=bmask, other=0).to(tl.int64)
        blk_swa = tl.load(map_ptr + blk_full, mask=bmask, other=-1).to(tl.int32)
        val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
        tl.store(swa_out + q * TW + k, val.to(tl.int32), mask=kmask)
        tl.store(topk_out + q, (cl + QO).to(tl.int32))


def block_meta(
    *,
    seq_lens,
    req_pool_indices,
    qo_len,
    req_to_token,
    mapping,
    out_loc,
    valid_from,
    window,
    width,
    max_seq_len,
    page_size,
    counter=None,
):
    """init_forward_metadata_dspark_draft_block's tensors in one launch: block_rows', the
    DSpark SWA page indices [rows, width] and lengths, and the page table (max_seq_len); with
    ``counter`` (int64 [1]) one added to it."""
    bs = seq_lens.shape[0]
    n = bs * qo_len
    dev = seq_lens.device
    i32 = {"dtype": torch.int32, "device": dev}
    num_cols = max(0, (min(max_seq_len, req_to_token.shape[1]) + page_size - 1) // page_size)
    out = {
        "seq_lens_casual": torch.empty(n, **i32),
        "req_pool_indices_repeated": torch.empty(n, dtype=req_pool_indices.dtype, device=dev),
        "positions_casual": torch.empty(n, **i32),
        "swa_page_indices": torch.empty((n, width), **i32),
        "swa_topk_lengths": torch.empty(n, **i32),
        "page_table": torch.empty((n, num_cols), **i32),
    }
    if n:
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        _block_meta_kernel[(n, max(1, triton.cdiv(num_cols, block)))](
            seq_lens,
            req_pool_indices,
            req_to_token,
            mapping,
            out_loc,
            valid_from if valid_from is not None else seq_lens,
            counter if counter is not None else seq_lens,
            out["seq_lens_casual"],
            out["req_pool_indices_repeated"],
            out["positions_casual"],
            out["swa_page_indices"],
            out["swa_topk_lengths"],
            out["page_table"],
            req_to_token.stride(0),
            num_cols,
            QO=qo_len,
            W=window,
            TW=width,
            TW_BLOCK=triton.next_power_of_2(width),
            PAGE=page_size,
            HAS_VALID=valid_from is not None,
            COUNT=counter is not None,
            BLOCK=block,
        )
    return out


@triton.jit
def _hc_mean_kernel(x_ptr, out_ptr, H, x_stride, out_stride, HC: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = h < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(HC):
        acc += tl.load(x_ptr + m * x_stride + k * H + h, mask=live, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + m * out_stride + h,
        (acc * (1.0 / HC)).to(out_ptr.dtype.element_ty),
        mask=live,
    )


def hc_mean_into(x, out):
    """``out[:] = x.mean(dim=1)`` for a contiguous [M, HC, H] bf16 ``x`` into a [M, H] view
    with unit inner stride (a column slice of the packed DSpark aux hidden states)."""
    m, hc, h = x.shape
    if m:
        block = 1024
        _hc_mean_kernel[(m, triton.cdiv(h, block))](x, out, h, x.stride(0), out.stride(0), HC=hc, BLOCK=block)
    return out


@triton.jit
def _post_combine_kernel(
    X,
    R,
    POST,
    COMB,
    PRE,
    OUT,
    Y,
    H,
    x_stride,
    r_stride,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = h < H
    x = tl.load(X + m * x_stride + h, mask=live, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(HC):
        # aiter's mhc_post as its compiler contracts it: x * post[i] fused onto residual[0] *
        # comb[0][i], then each further stream's product fused on in stream order
        r0 = tl.load(R + m * r_stride + h, mask=live, other=0.0).to(tl.float32)
        v = tl.fma(x, tl.load(POST + m * HC + i), r0 * tl.load(COMB + m * HC * HC + i))
        for j in tl.static_range(1, HC):
            r = tl.load(R + m * r_stride + j * H + h, mask=live, other=0.0).to(tl.float32)
            v = tl.fma(r, tl.load(COMB + m * HC * HC + j * HC + i), v)
        vb = v.to(tl.bfloat16)
        tl.store(OUT + m * HC * H + i * H + h, vb, mask=live)
        # sglang's mhc_combine of the rounded stream, as its kernel writes it
        pk = tl.load(PRE + m * HC + i).to(tl.float32)
        acc += pk * vb.to(tl.float32)
    tl.store(Y + m * H + h, acc.to(tl.bfloat16), mask=live)


def post_combine(x, residual, post, comb, pre):
    """aiter's ``mhc_post(x, residual, post, comb)`` (the streams [M, HC, H] bf16) and sglang's
    ``mhc_combine`` of them by ``pre`` ([M, H] bf16) in one launch."""
    m, hc, h = residual.shape
    out = torch.empty((m, hc, h), dtype=residual.dtype, device=residual.device)
    y = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    if m:
        block = 1024
        _post_combine_kernel[(m, triton.cdiv(h, block))](
            x,
            residual,
            post,
            comb,
            pre,
            out,
            y,
            h,
            x.stride(0),
            residual.stride(0),
            HC=hc,
            BLOCK=block,
        )
    return out, y


def metadata_disagrees(want, got, path="metadata", seen=None) -> str | None:
    """The first field where ``got`` differs from ``want`` (tensors by dtype, shape and
    value; dataclasses, structs, dicts and sequences by their members), or None."""
    import dataclasses

    seen = set() if seen is None else seen
    if id(want) in seen:
        return None
    if isinstance(want, torch.Tensor) or isinstance(got, torch.Tensor):
        if not (isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor)):
            return f"{path}: {type(got).__name__} vs sglang's {type(want).__name__}"
        if want.dtype != got.dtype or want.shape != got.shape:
            return f"{path}: {got.dtype}{tuple(got.shape)} vs sglang's {want.dtype}{tuple(want.shape)}"
        if any(f".{name}" in path for name in _UNWRITTEN):
            return None
        if want.numel() and not torch.equal(want, got):
            return f"{path}: {(want != got).sum().item()} of {want.numel()} values differ"
        return None
    if type(want) is not type(got):
        return f"{path}: {type(got).__name__} vs sglang's {type(want).__name__}"
    seen.add(id(want))
    if dataclasses.is_dataclass(want):
        names = [f.name for f in dataclasses.fields(want)]
    elif hasattr(type(want), "__struct_fields__"):
        names = list(type(want).__struct_fields__)
    elif isinstance(want, dict):
        if want.keys() != got.keys():
            return f"{path}: keys {sorted(map(str, got))} vs sglang's {sorted(map(str, want))}"
        for k in want:
            why = metadata_disagrees(want[k], got[k], f"{path}[{k!r}]", seen)
            if why:
                return why
        return None
    elif isinstance(want, (list, tuple)):
        if len(want) != len(got):
            return f"{path}: {len(got)} items vs sglang's {len(want)}"
        for i, (a, b) in enumerate(zip(want, got)):
            why = metadata_disagrees(a, b, f"{path}[{i}]", seen)
            if why:
                return why
        return None
    else:
        return (
            None
            if want == got or not isinstance(want, (int, float, str, bool))
            else (f"{path}: {got!r} vs sglang's {want!r}")
        )
    for name in names:
        why = metadata_disagrees(getattr(want, name), getattr(got, name), f"{path}.{name}", seen)
        if why:
            return why
    return None


def expand_disagrees(orig, backend, bs: int, qo_len: int, seq_lens, req_pool_indices):
    """Why ``expand`` cannot stand in for ``orig`` on these inputs, or None."""
    want = orig(backend, bs, qo_len, seq_lens, req_pool_indices)
    got = expand(bs, qo_len, seq_lens, req_pool_indices)
    for name, a, b in zip(("seq_lens_casual", "req_pool_indices_repeated"), got, want):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return f"{name}: {a.dtype}{tuple(a.shape)} vs sglang's {b.dtype}{tuple(b.shape)}"
    return None


def swa_disagrees(want: torch.Tensor, got: torch.Tensor) -> str | None:
    if want.dtype != got.dtype or want.shape != got.shape:
        return f"{got.dtype}{tuple(got.shape)} vs sglang's {want.dtype}{tuple(want.shape)}"
    if not torch.equal(want, got):
        return f"{(want != got).sum().item()} of {want.numel()} indices differ"
    return None
