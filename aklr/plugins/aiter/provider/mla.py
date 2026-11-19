# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MLA decode ops, served by AKLR's ``aklr.ops.mla``: ``mla_decode_fwd`` (aiter's ASM decode)
by ``mla_decode_fwd``, ``mla_paged_decode`` (the decode mode of aiter's ``mla_gluon``) by
``flash_mla_with_kvcache``'s dense pool, and ``fp8_paged_mqa_logits`` (aiter's
``deepgemm_fp8_paged_mqa_logits``) by ``fp8_paged_mqa_logits``, or, for the rows of a target verify a
framework noted with :func:`note_target_verify`, by ``fp8_paged_mqa_logits_verify``."""

from __future__ import annotations

import torch

from aklr.ops.mla.api import (
    flash_mla_with_kvcache,
    fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_verify,
    mla_decode_fwd,
    mla_decode_fwd_cprr,
)
from aklr.plugins.aiter.provider._base import offer, selector, served

__all__ = ["note_target_verify"]

_LATENT = 512  # the no-rope part of q and of a pool row: the values read
_KV_CACHE_FORMATS = {torch.bfloat16: "bf16", torch.float8_e4m3fn: "fp8_e4m3"}


def _sm_scale(q, sm_scale):
    """aiter's default: one over the root of the 576 values a logit sums over."""
    return q.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)


# mla_decode_fwd's keyword-only parameters, which only its NVFP4 path takes
_NVFP4 = ("q_aux", "kv_aux", "nvfp4_params", "nvfp4_lse")


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
        cp_rank,
        causal,
        kv_layout,
        seq_lens,
        page_table,
        kv_csr_layout,
        _fast_mode,
    ) = args
    # a page-size-1 pool of token-major rows, its lse returned. aiter's schedule (work and reduce maps,
    # splits, batch and fast modes) only orders its own kernel's work. The keyword-only tensors are
    # aiter's NVFP4 path
    if (
        any(kwargs[name] is not None for name in _NVFP4)
        or page_size != 1
        or nhead_kv != 1
        or logit_cap > 0
        or num_kv_splits_indptr is not None
        or return_logits
        or not return_lse
        or q_scale is None
        or kv_scale is None
        or kv_layout != "token_major"
        or seq_lens is not None
        or page_table is not None
        or kv_csr_layout != "page"
        or q.dim() != 3
    ):
        return None
    batch = kv_indptr.shape[0] - 1
    sm_scale = _sm_scale(q, sm_scale)
    # one query a request: causal has no effect on it, nor cp_rank without context parallelism
    if g_kv_indptr is None and cp_world_size == 1:
        if max_seqlen_q != 1 or q.shape[0] != batch:
            return None
        return offer(
            mla_decode_fwd.serving(
                q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=sm_scale, o=o
            ),
            _run_mla_decode_fwd,
        )
    # vLLM's verify of draft tokens under decode context parallelism: as many causal queries a request
    # (rows of max_seqlen_q a request, so none fewer) over its round-robin shard of the pool
    cprr = _cprr_call(
        q,
        kv_buffer,
        o,
        kv_indptr,
        kv_indices,
        max_seqlen_q,
        sm_scale,
        q_scale,
        kv_scale,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        causal,
    )
    if cprr is None:
        return None
    call_args, call_kwargs = cprr
    return offer(mla_decode_fwd_cprr.serving(*call_args, **call_kwargs), _run_mla_decode_fwd)


def _cprr_call(
    q,
    kv_buffer,
    o,
    kv_indptr,
    kv_indices,
    max_seqlen_q,
    sm_scale,
    q_scale,
    kv_scale,
    g_kv_indptr,
    cp_world_size,
    cp_rank,
    causal,
):
    """``mla_decode_fwd_cprr``'s (args, kwargs) for aiter's call, its q and o as [batch, queries, heads,
    width]; ``None`` outside it."""
    batch = kv_indptr.shape[0] - 1
    if (
        g_kv_indptr is None
        or cp_world_size < 2
        or not causal
        or max_seqlen_q < 2
        or q.shape[0] != batch * max_seqlen_q
        or o.shape[0] != q.shape[0]
    ):
        return None
    queries = (batch, max_seqlen_q)
    return (q.unflatten(0, queries), kv_buffer, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale), {
        "cp_world_size": cp_world_size,
        "sm_scale": sm_scale,
        "cp_rank": cp_rank,
        "o": o.unflatten(0, queries),
    }


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
    *,
    q_aux=None,
    kv_aux=None,
    nvfp4_params=None,
    nvfp4_lse=None,
):
    sm_scale = _sm_scale(q, sm_scale)
    cprr = _cprr_call(
        q,
        kv_buffer,
        o,
        kv_indptr,
        kv_indices,
        max_seqlen_q,
        sm_scale,
        q_scale,
        kv_scale,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        causal,
    )
    # (o, lse): aiter returns its split scratch first, which its callers do not read
    if cprr is None:
        return served(
            mla_decode_fwd(q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=sm_scale, o=o)
        )
    call_args, call_kwargs = cprr
    _, lse = served(mla_decode_fwd_cprr(*call_args, **call_kwargs))
    # aiter's lse is a row's per head, as its q's rows
    return o, lse.flatten(0, 1)


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
    # AKLR's api takes no output: aiter's caller reads its o. aiter's op returns lse alone, an empty fp32
    # tensor when return_lse is off (a custom op cannot return None for a tensor)
    o.copy_(output.view(o.shape))
    return lse if return_lse else torch.empty(0, dtype=torch.float32, device=o.device)


def _context_lens(context_lens):
    """aiter's per-request lengths, [B] or [B, 1], as AKLR's [B]; ``None`` for any other layout."""
    if context_lens.dim() == 2 and context_lens.shape[1] == 1:
        return context_lens.squeeze(1)
    return context_lens if context_lens.dim() == 1 else None


# (query rows, draft tokens a request) of the target verify whose logits the framework asks next. Under
# decode context parallelism sglang's indexer hands a verify's rows to fp8_paged_mqa_logits as rows of
# their own, each with its request's block table row, as a decode batch's rows come, so only the
# framework's forward mode tells the two apart. Select only reads it (aiter's explain() asks select too and
# runs nothing); the call that runs takes it, and the framework's next paged top-k notes it afresh.
_TARGET_VERIFY: list[tuple[int, int] | None] = [None]


def note_target_verify(rows: int, draft_tokens: int) -> None:
    """Note that the next fp8_paged_mqa_logits call, if over ``rows`` query rows, is a target verify's:
    ``draft_tokens`` consecutive rows a request, each passed the request's block table row. AKLR then
    scores each request's rows together with ``fp8_paged_mqa_logits_verify``. ``rows`` 0 clears it."""
    _TARGET_VERIFY[0] = (rows, draft_tokens) if rows else None


def _verify_call(q_fp8, kv_cache, weights, lens, kv_indices, out_logits, page_size, draft_tokens):
    """fp8_paged_mqa_logits_verify's keywords but the block tables, and the rows of kv_indices they are,
    for a call of ``draft_tokens`` rows a request; ``None`` when the rows cannot be regrouped by views."""
    rows = q_fp8.shape[0]
    if draft_tokens < 2 or rows == 0 or rows % draft_tokens:
        return None
    if not all(t.is_contiguous() for t in (q_fp8, weights, lens, out_logits)):
        return None
    batch = rows // draft_tokens
    # each request's first row: its other rows repeat it, and table rows past the queries' are not read
    return kv_indices[:rows:draft_tokens], {
        "q": q_fp8.view(batch, draft_tokens, *q_fp8.shape[2:]),
        "kv_cache": kv_cache.view(torch.uint8),
        "weights": weights.view(batch, draft_tokens, -1),
        "context_lens": lens.view(batch, draft_tokens),
        "logits": out_logits.view(batch, draft_tokens, -1),
        "page_size": page_size,
        "preshuffled": True,
    }


def _select_target_verify(q_fp8, kv_cache, weights, lens, kv_indices, out_logits, page_size, note):
    """The candidate for the rows of the target verify noted, else ``None``."""
    if note is None or note[0] != q_fp8.shape[0]:
        return None
    call = _verify_call(q_fp8, kv_cache, weights, lens, kv_indices, out_logits, page_size, note[1])
    if call is None:
        return None
    tables, kwargs = call
    # asked with a contiguous stand-in: the rows are copied only for a call AKLR takes
    stand_in = torch.empty(tables.shape, dtype=tables.dtype, device=tables.device)
    return offer(fp8_paged_mqa_logits_verify.serving(block_tables=stand_in, **kwargs), _verify_run(note[1]))


def _verify_run(draft_tokens):
    """The run of a target verify's call, ``draft_tokens`` rows a request; aiter's parameters exactly."""

    def run(
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
        _TARGET_VERIFY[0] = None
        tables, kwargs = _verify_call(
            q_fp8,
            kv_cache,
            weights,
            _context_lens(context_lens),
            kv_indices,
            out_logits,
            KVBlockSize,
            draft_tokens,
        )
        # writes out_logits through its view, as aiter does
        served(fp8_paged_mqa_logits_verify(block_tables=tables.contiguous(), **kwargs))

    return run


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
    note = _TARGET_VERIFY[0]
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
    verify = _select_target_verify(q_fp8, kv_cache, weights, lens, kv_indices, out_logits, KVBlockSize, note)
    if verify is not None:
        return verify
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
    _TARGET_VERIFY[0] = None
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
