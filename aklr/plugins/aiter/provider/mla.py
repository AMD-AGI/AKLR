# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MLA decode ops, served by AKLR's ``aklr.ops.mla``: ``mla_decode_fwd`` (aiter's ASM decode)
by ``mla_decode_fwd``, ``mla_paged_decode`` (the decode mode of aiter's ``mla_gluon``) by
``flash_mla_with_kvcache``'s dense pool, and ``fp8_paged_mqa_logits`` (aiter's
``deepgemm_fp8_paged_mqa_logits``) by ``fp8_paged_mqa_logits``."""

from __future__ import annotations

import torch

from aklr.ops.mla.api import flash_mla_with_kvcache, fp8_paged_mqa_logits, mla_decode_fwd
from aklr.plugins.aiter.provider._base import offer, selector, served

_LATENT = 512  # the no-rope part of q and of a pool row: the values read
_KV_CACHE_FORMATS = {torch.bfloat16: "bf16", torch.float8_e4m3fn: "fp8_e4m3"}


def _sm_scale(q, sm_scale):
    """aiter's default: one over the root of the 576 values a logit sums over."""
    return q.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)


@selector("mla_decode_fwd")
def _select_mla_decode_fwd(args, kwargs):
    (
        q,
        kv_buffer,
        o,
        _qo_indptr,
        kv_indptr,
        kv_indices,
        _kv_last_page_lens,
        max_seqlen_q,
        page_size,
        nhead_kv,
        sm_scale,
        logit_cap,
        _num_kv_splits,
        num_kv_splits_indptr,
        _work_meta_data,
        _work_indptr,
        _work_info_set,
        _reduce_indptr,
        _reduce_final_map,
        _reduce_partial_map,
        q_scale,
        kv_scale,
        _intra_batch_mode,
        return_logits,
        return_lse,
        g_kv_indptr,
        cp_world_size,
        _cp_rank,
        _causal,
        kv_layout,
        seq_lens,
        page_table,
        kv_csr_layout,
        _fast_mode,
    ) = args
    # one query a request over a page-size-1 pool of token-major rows, its lse returned. aiter's
    # schedule (work and reduce maps, splits, batch and fast modes) only orders its own kernel's work;
    # causal has no effect on one query, nor cp_rank without context parallelism
    if (
        max_seqlen_q != 1
        or page_size != 1
        or nhead_kv != 1
        or logit_cap > 0
        or num_kv_splits_indptr is not None
        or return_logits
        or not return_lse
        or q_scale is None
        or kv_scale is None
        or g_kv_indptr is not None
        or cp_world_size != 1
        or kv_layout != "token_major"
        or seq_lens is not None
        or page_table is not None
        or kv_csr_layout != "page"
        or q.dim() != 3
        or q.shape[0] != kv_indptr.shape[0] - 1
    ):
        return None
    return offer(
        mla_decode_fwd.serving(
            q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=_sm_scale(q, sm_scale), o=o
        ),
        _run_mla_decode_fwd,
    )


def _run_mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size,
    nhead_kv,
    sm_scale,
    logit_cap,
    num_kv_splits,
    num_kv_splits_indptr,
    work_meta_data,
    work_indptr,
    work_info_set,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    q_scale,
    kv_scale,
    intra_batch_mode,
    return_logits,
    return_lse,
    g_kv_indptr,
    cp_world_size,
    cp_rank,
    causal,
    kv_layout,
    seq_lens,
    page_table,
    kv_csr_layout,
    fast_mode,
):
    # (o, lse): aiter returns its split scratch first, which its callers do not read
    return served(
        mla_decode_fwd(
            q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=_sm_scale(q, sm_scale), o=o
        )
    )


def _whole_query(q_nope, q_pe):
    """The q of rows of latent then rope values whose two adjacent views the caller passes as q_nope
    and q_pe; ``None`` for two tensors apart."""
    lead, latent = q_nope.shape[:-1], q_nope.shape[-1]
    width = latent + q_pe.shape[-1]
    strides, step = [1], width
    for size in reversed(lead):
        strides.insert(0, step)
        step *= size
    strides = tuple(strides)
    if not (
        q_pe.dtype == q_nope.dtype
        and q_pe.shape[:-1] == lead
        and q_nope.stride() == q_pe.stride() == strides
        and q_pe.untyped_storage().data_ptr() == q_nope.untyped_storage().data_ptr()
        and q_pe.storage_offset() == q_nope.storage_offset() + latent
    ):
        return None
    return q_nope.as_strided((*lead, width), strides)


def _flash_mla_call(
    q_nope, q_pe, kv_c, page_table, seq_info, sm_scale, k_pe, kv_pe_offset, use_2d_view, kv_scale
):
    """AKLR's dense ``flash_mla_with_kvcache`` call for aiter's paged MLA decode, as (args, kwargs);
    ``None`` outside it. aiter's 4-D q is several causal queries a request."""
    kv_cache_format = _KV_CACHE_FORMATS.get(kv_c.dtype)
    # the rope part in the pool rows themselves, right after the latent
    if (
        k_pe is not None
        or kv_pe_offset != _LATENT
        or q_nope.shape[-1] != _LATENT
        or q_nope.dim() not in (3, 4)
        or kv_c.dim() != 2
        or kv_cache_format is None
    ):
        return None
    q = _whole_query(q_nope, q_pe)
    if q is None or kv_c.shape[-1] != q.shape[-1]:
        return None
    is_mtp = q.dim() == 4
    return (q if is_mtp else q.unsqueeze(1), kv_c[:, None, None, :]), {
        "block_table": page_table,
        "cache_seqlens": seq_info if use_2d_view else None,
        "cu_seqlens_k": None if use_2d_view else seq_info,
        "causal": is_mtp,
        "head_dim_v": _LATENT,
        "sm_scale": sm_scale,
        "kv_cache_format": kv_cache_format,
        "kv_scale": kv_scale,
        "return_lse": True,
    }


@selector("mla_paged_decode")
def _select_mla_paged_decode(args, kwargs):
    # min_kv_seq_len only promises aiter's kernel a lower bound; return_lse only drops the lse
    (
        q_nope,
        q_pe,
        kv_c,
        o,
        page_table,
        seq_info,
        sm_scale,
        k_pe,
        kv_pe_offset,
        use_2d_view,
        kv_scale,
        _min_kv_seq_len,
        _return_lse,
    ) = args
    call = _flash_mla_call(
        q_nope, q_pe, kv_c, page_table, seq_info, sm_scale, k_pe, kv_pe_offset, use_2d_view, kv_scale
    )
    # AKLR's output is copied into o: one shaped and typed as aiter would write it
    if call is None or o.shape != q_nope.shape or o.dtype != q_nope.dtype:
        return None
    call_args, call_kwargs = call
    return offer(flash_mla_with_kvcache.serving(*call_args, **call_kwargs), _run_mla_paged_decode)


def _run_mla_paged_decode(
    q_nope,
    q_pe,
    kv_c,
    o,
    page_table,
    seq_info,
    sm_scale,
    k_pe,
    kv_pe_offset,
    use_2d_view,
    kv_scale,
    min_kv_seq_len,
    return_lse,
):
    call_args, call_kwargs = _flash_mla_call(
        q_nope, q_pe, kv_c, page_table, seq_info, sm_scale, k_pe, kv_pe_offset, use_2d_view, kv_scale
    )
    output, lse = served(flash_mla_with_kvcache(*call_args, **call_kwargs))
    # AKLR's api takes no output: aiter's caller reads its o
    o.copy_(output.view(o.shape))
    return o, (lse if return_lse else None)


def _context_lens(context_lens):
    """aiter's per-request lengths, [B] or [B, 1], as AKLR's [B]; ``None`` for any other layout."""
    if context_lens.dim() == 2 and context_lens.shape[1] == 1:
        return context_lens.squeeze(1)
    return context_lens if context_lens.dim() == 1 else None


@selector("fp8_paged_mqa_logits")
def _select_fp8_paged_mqa_logits(args, kwargs):
    (
        q_fp8,
        kv_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        max_model_len,
        Preshuffle,
        KVBlockSize,
        _ChunkK,
        _TotalCuCount,
        _WavePerEU,
        VarCtxSchedule,
        _SplitKV,
        PersistentSchedule,
    ) = args
    lens = _context_lens(context_lens)
    # one query a request into whole pages of preshuffled keys, AKLR's logits as wide as the block
    # table and as the bound aiter stores within; a precomputed schedule carries lengths of its own.
    # ChunkK, TotalCuCount, WavePerEU and SplitKV only size aiter's grid
    if (
        lens is None
        or q_fp8.dim() != 4
        or q_fp8.shape[1] != 1
        or not Preshuffle
        or VarCtxSchedule is not None
        or PersistentSchedule is not None
        or kv_cache.element_size() != 1
        or out_logits.dim() != 2
        or kv_indices.dim() != 2
        or out_logits.shape[1] != kv_indices.shape[1] * KVBlockSize
        or max_model_len != out_logits.shape[1]
    ):
        return None
    return offer(
        fp8_paged_mqa_logits.serving(
            q_fp8,
            kv_cache.view(torch.uint8),
            weights,
            lens,
            kv_indices,
            out_logits,
            page_size=KVBlockSize,
            preshuffled=True,
        ),
        _run_fp8_paged_mqa_logits,
    )


def _run_fp8_paged_mqa_logits(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    Preshuffle,
    KVBlockSize,
    ChunkK,
    TotalCuCount,
    WavePerEU,
    VarCtxSchedule,
    SplitKV,
    PersistentSchedule,
):
    served(
        fp8_paged_mqa_logits(
            q_fp8,
            kv_cache.view(torch.uint8),
            weights,
            _context_lens(context_lens),
            kv_indices,
            out_logits,
            page_size=KVBlockSize,
            preshuffled=True,
        )
    )
    # aiter writes out_logits and returns nothing
