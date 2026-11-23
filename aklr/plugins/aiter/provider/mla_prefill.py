# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MLA prefill ops AKLR serves:

- ``mla_sparse_prefill_v4`` (also reached from ``pa_sparse_prefill_fp8_opus``), by AKLR's
  ``sparse_prefill_mxfp8``, whose baseline is that aiter kernel: the same eleven tensors in the same
  order, then the softmax scale.
- ``flash_attn_varlen`` with its lse, the pass vLLM's ``ROCM_AITER_FA`` MLA prefill backend makes over
  a step's new tokens and over each context chunk, by AKLR's ``mla_prefill_varlen_lse`` in fp8 (q, k
  and v rounded to float8_e4m3fn) instead of aiter's bf16 flash attention."""

from __future__ import annotations

import torch

from aklr.ops.mla_prefill.api import mla_prefill_varlen_lse, sparse_prefill_mxfp8
from aklr.plugins.aiter.provider._base import offer, selector, served

# Fewer query rows stay on aiter: a decode step's extends (a few tokens over the whole context) reach
# flash_attn_varlen too, and AKLR's fp8 kernel, built for prefill chunks, casts the whole context and
# plans per shape on each call. Its workloads start at 257 rows.
_MIN_ROWS = 256


def _sparse_prefill_call(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
):
    """AKLR's call, as (args, kwargs); ``None`` outside it. AKLR reads the packed NoPE rows (448 e4m3fn
    codes, their E8M0 scales, padding) as bytes, and always takes a sink."""
    nope = (q_nope, unified_kv_nope, kv_nope)
    if attn_sink is None or any(t.dtype != torch.float8_e4m3fn for t in nope):
        return None
    q_nope, unified_kv_nope, kv_nope = (t.view(torch.uint8) for t in nope)
    return (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ), {"softmax_scale": softmax_scale}


@selector("mla_sparse_prefill_v4")
def _select_mla_sparse_prefill_v4(args, kwargs):
    # its second tensor is q_rope (mla_sparse_prefill_v41 takes q_scale there); num_kv_splits only
    # splits aiter's own kernel's work
    (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        _num_kv_splits,
    ) = args
    # AKLR's output is copied into out: one aiter takes, [T, H, 512] bf16, else aiter raises
    if out is not None and (out.shape != q_nope.shape or out.dtype != torch.bfloat16):
        return None
    call = _sparse_prefill_call(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
    if call is None:
        return None
    call_args, call_kwargs = call
    return offer(sparse_prefill_mxfp8.serving(*call_args, **call_kwargs), _run_mla_sparse_prefill_v4)


def _run_mla_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits,
):
    call_args, call_kwargs = _sparse_prefill_call(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
    output = served(sparse_prefill_mxfp8(*call_args, **call_kwargs))
    # AKLR's api takes no output: aiter writes the caller's out when given and returns it either way
    if out is None:
        return output
    out.copy_(output)
    return out


def _aligned(offsets, copy):
    """``offsets``, or where its data does not start on the 16 bytes AKLR's dispatch requires, a copy
    (``copy``) or an aligned tensor of its shape to ask with, which enqueues nothing."""
    if offsets is None or offsets.data_ptr() % 16 == 0:
        return offsets
    return offsets.clone() if copy else torch.empty_like(offsets)


def _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy):
    """mla_prefill_varlen_lse's offsets and keywords for aiter's call, unaligned offsets copied when
    ``copy`` (to run) or stood in for (to ask). Both causal masks are aligned at the end of the keys;
    keys that are the queries' own rows are AKLR's default."""
    keys = None if cu_seqlens_k is cu_seqlens_q else _aligned(cu_seqlens_k, copy)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return _aligned(cu_seqlens_q, copy), {
        "qkv_fp8": True,
        "causal": bool(causal),
        "cu_seqlens_k": keys,
        "sm_scale": scale,
    }


@selector("flash_attn_varlen")
def _select_flash_attn_varlen(args, kwargs):
    # min_seqlen_q and how_v3_bf16_cvt tune aiter's kernels, max_seqlen_* size its grid, and the forward
    # pass is deterministic either way
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        _max_seqlen_q,
        _max_seqlen_k,
        _min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        bias,
        alibi_slopes,
        _deterministic,
        return_lse,
        return_attn_probs,
        _how_v3_bf16_cvt,
        block_table,
        out,
        cu_seqlens_q_padded,
        cu_seqlens_k_padded,
        sink_ptr,
        q_descale,
        k_descale,
        v_descale,
    ) = args
    # AKLR takes the pass vLLM merges over cached context: (output, lse), bf16 rows, no mask but the
    # causal one, and an output of its own
    if not return_lse or return_attn_probs or dropout_p or logits_soft_cap:
        return None
    window = tuple(window_size)
    if window[:2] != (-1, -1) or any(window[2:]):
        return None
    extras = (bias, alibi_slopes, block_table, out, cu_seqlens_q_padded, cu_seqlens_k_padded, sink_ptr)
    if any(t is not None for t in (*extras, q_descale, k_descale, v_descale)):
        return None
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or q.shape[0] < _MIN_ROWS:
        return None
    # AKLR's result has no autograd graph; aiter's own path keeps the backward
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return None
    offsets, call = _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy=False)
    # v may be a strided view (vLLM's is a slice of kv_b's output): asked with a contiguous stand-in
    stand_in = v if v.is_contiguous() else torch.empty(v.shape, dtype=v.dtype, device=v.device)
    return offer(mla_prefill_varlen_lse.serving(q, k, stand_in, offsets, **call), _run_flash_attn_varlen)


def _run_flash_attn_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q,
    dropout_p,
    softmax_scale,
    logits_soft_cap,
    causal,
    window_size,
    bias,
    alibi_slopes,
    deterministic,
    return_lse,
    return_attn_probs,
    how_v3_bf16_cvt,
    block_table,
    out,
    cu_seqlens_q_padded,
    cu_seqlens_k_padded,
    sink_ptr,
    q_descale,
    k_descale,
    v_descale,
):
    offsets, call = _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy=True)
    output, lse = served(mla_prefill_varlen_lse(q, k, v.contiguous(), offsets, **call))
    # aiter's lse is [heads, tokens], AKLR's [tokens, heads]
    return output, lse.t().contiguous()
