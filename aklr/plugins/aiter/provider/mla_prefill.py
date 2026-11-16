# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MLA v4 sparse prefill, ``mla_sparse_prefill_v4`` (also reached from
``pa_sparse_prefill_fp8_opus``), served by AKLR's ``sparse_prefill_mxfp8``, whose baseline is that
aiter kernel: the same eleven tensors in the same order, then the softmax scale."""

from __future__ import annotations

import torch

from aklr.ops.mla_prefill.api import sparse_prefill_mxfp8
from aklr.plugins.aiter.provider._base import offer, selector, served


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
