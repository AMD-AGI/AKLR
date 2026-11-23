# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Select and run paged MQA logits for decode and target verify."""

import torch

from aklr.plugins.aiter import _verify
from aklr.plugins.aiter._candidate import offer, served

from . import plugin as ops


def _context_lens(context_lens):
    """aiter's per-request lengths, [B] or [B, 1], as AKLR's [B]; ``None`` for any other layout."""
    if context_lens.dim() == 2 and context_lens.shape[1] == 1:
        return context_lens.squeeze(1)
    return context_lens if context_lens.dim() == 1 else None


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
    return offer(
        ops.fp8_paged_mqa_logits_verify.serving(block_tables=stand_in, **kwargs), _verify_run(note[1])
    )


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
        _verify.consume()
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
        served(ops.fp8_paged_mqa_logits_verify(block_tables=tables.contiguous(), **kwargs))

    return run


def select(args, kwargs):
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
    note = _verify.current()
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
        ops.fp8_paged_mqa_logits.serving(
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
    _verify.consume()
    served(
        ops.fp8_paged_mqa_logits(
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
