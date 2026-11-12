# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA prefill sglang's aiter backend runs on aiter's fp8 path (``srt/layers/attention/aiter_backend.py``):
AKLR's ``mla_prefill_varlen`` op, and ``mla_prefill_varlen_lse`` for the passes that need each query's
log-sum-exp.

sglang plans aiter's work once a forward, then at each MLA layer tiles the heads up to one aiter's
reduce instantiates, casts q, k and v to fp8 and runs aiter's ASM prefill and reduce. The route
takes the layer's call whole: the offsets from the plan, the rest from the call.

A chunked prefix runs two kinds of passes that return (output, lse) for merge_state: the causal pass over
the extend tokens and, per prefix chunk, the pass every query reads whole. Both go to ``mla_prefill_varlen_lse``.
Where AKLR declines one of them, it runs on aiter's bf16 flash attention with return_lse (the kernel sglang
uses for these passes without a PS plan), never on aiter's fp8 ASM prefill, which faulted in sglang's
200K-token prefill on the first batch past the 131072-key chunk capacity (prefix 116352, extend 16384); each
such fallback is logged. Since neither reads aiter's PS plan, the plans sglang builds for these passes
(need_lse) are left unbuilt: aiter's planner and sglang's check of its partial rows, which failed in a 200K
prefill (row 131072 of 16384), are skipped."""

from __future__ import annotations

import logging

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla_prefill.api import mla_prefill_varlen, mla_prefill_varlen_lse
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []

_LOGGER = logging.getLogger(__name__)
_KV_SEGMENT = 65536  # the most keys one ASM call of mla_prefill_varlen_lse's kernel reads
_LOGGED: set[tuple] = set()


def _log_once(key: tuple, message: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _LOGGER.warning(message, *args)


def _flash_with_lse(q, k, v, layer, ps):
    """(output, lse) of the pass over bf16 flash attention, lse as merge_state reads it: [tokens, heads]."""
    from aiter import flash_attn_varlen_func

    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        ps.qo_indptr,
        ps.kv_indptr,
        ps.max_q_len,
        # the key rows bound the longest sequence's keys without reading the offsets
        k.shape[0],
        softmax_scale=layer.scaling,
        causal=ps.is_causal,
        return_lse=True,
    )[:2]
    return output, lse.transpose(0, 1).contiguous()


@sglang_route(
    "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend._build_prefill_ps_metadata",
    versions=("0.5.22",),
)
def _build_prefill_ps_metadata(
    self,
    *,
    qo_indptr,
    kv_indptr,
    kv_lens_cpu,
    num_kv_tokens,
    max_q_len,
    is_causal,
    need_lse,
    qo_indptr_cpu=None,
    kv_indptr_cpu=None,
    exact_partial_count=True,
):
    # without need_lse the plan is the main prefill's, whose route may hand the call back to aiter's fp8 path
    if not need_lse:
        return FallbackReason.OUT_OF_CONTRACT
    from sglang.srt.layers.attention.aiter_backend import MlaPrefillPsMetadata

    # what _lse_pass reads; aiter's plan buffers stay None, read by no path these passes take
    return MlaPrefillPsMetadata(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=None,
        work_metadata=None,
        work_indptr=None,
        work_info_set=None,
        reduce_indptr=None,
        reduce_final_map=None,
        reduce_partial_map=None,
        max_q_len=max_q_len,
        is_causal=is_causal,
        need_lse=True,
        num_partial_tiles=0,
    )


def _aligned(offsets):
    """offsets, or a copy where its data does not start on the 16 bytes AKLR's dispatch requires: sglang's
    prefix_chunk_cu_seq_lens is [chunks, sequences + 1] int32, so with one sequence every other chunk's row starts
    8 bytes past a boundary."""
    return offsets if offsets is None or offsets.data_ptr() % 16 == 0 else offsets.clone()


def _lse_pass(q, k, v, layer, ps):
    """(output, lse) of a chunked-prefix pass on mla_prefill_varlen_lse, else on bf16 flash attention."""
    # a prefix chunk's keys are not the queries' rows even when as many; the causal pass reads the queries' own
    offsets_k = None if ps.is_causal and k.shape[0] == q.shape[0] else _aligned(ps.kv_indptr)
    offsets_q = _aligned(ps.qo_indptr)
    stand_in = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    args = {"qkv_fp8": True, "causal": ps.is_causal, "cu_seqlens_k": offsets_k, "sm_scale": layer.scaling}
    reason = mla_prefill_varlen_lse.fallback_reason(q, k, stand_in, offsets_q, **args)
    if reason is None:
        if ps.qo_indptr.shape[0] > 2:
            _log_once(
                ("several", ps.is_causal),
                "AKLR mla_prefill_varlen_lse: a batch of several sequences (causal=%s) runs on bf16 flash attention "
                "inside the AKLR kernel, not fp8",
                ps.is_causal,
            )
        result = mla_prefill_varlen_lse(q, k, v.contiguous(), offsets_q, **args)
        if not isinstance(result, FallbackReason):
            return result
        reason = result
    _log_once(
        (reason, ps.is_causal, q.shape[0], k.shape[0]),
        "AKLR mla_prefill_varlen_lse declined (%s) a chunked-prefix pass (causal=%s, %d queries, %d keys): it runs "
        "on aiter's bf16 flash attention",
        reason.value,
        ps.is_causal,
        q.shape[0],
        k.shape[0],
    )
    return _flash_with_lse(q, k, v, layer, ps)


@sglang_route(
    "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend._mla_fp8_prefill_attn_ps",
    versions=("0.5.22",),
)
def _mla_fp8_prefill_attn_ps(self, q, k, v, layer, ps):
    # the chunked prefix's passes need the LSE, and its prefix chunks attend without the causal mask; their plan
    # is unbuilt, so they never return to sglang
    if ps.need_lse or not ps.is_causal:
        output, lse = _lse_pass(q, k, v, layer, ps)
        return output, (lse if ps.need_lse else None)
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return FallbackReason.OUT_OF_CONTRACT
    # without a prefix the keys are the queries' own rows
    offsets_k = _aligned(ps.kv_indptr) if k.shape[0] != q.shape[0] else None
    offsets_q = _aligned(ps.qo_indptr)
    # past _KV_SEGMENT keys, mla_prefill_varlen_lse's kernel, whose every ASM call reads at most that many, its lse
    # unused
    if offsets_k is not None and k.shape[0] > _KV_SEGMENT:
        stand_in = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        args = {"qkv_fp8": True, "causal": True, "cu_seqlens_k": offsets_k, "sm_scale": layer.scaling}
        if mla_prefill_varlen_lse.fallback_reason(q, k, stand_in, offsets_q, **args) is None:
            result = mla_prefill_varlen_lse(q, k, v.contiguous(), offsets_q, **args)
            if not isinstance(result, FallbackReason):
                return result[0], None
    # kv_b's output leaves v a strided view: ask before copying it whole
    stand_in = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    args = {"qkv_fp8": True, "cu_seqlens_k": offsets_k, "sm_scale": layer.scaling}
    if (reason := mla_prefill_varlen.fallback_reason(q, k, stand_in, offsets_q, **args)) is not None:
        return reason
    output = mla_prefill_varlen(q, k, v.contiguous(), offsets_q, **args)
    return output if isinstance(output, FallbackReason) else (output, None)
