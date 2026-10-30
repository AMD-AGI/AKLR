# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["flash_mla_with_kvcache_baseline", "fp8_paged_mqa_logits_baseline"]


def flash_mla_with_kvcache_baseline(
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
    return_lse=False,
    kv_cache_layout="row_major",
    output=None,
    lse=None,
):
    """Sparse: sglang's kernel, as the route calls it. Dense: aiter's Gluon MLA decode
    over the 2-D block table, the kernel sglang serves the DSPARK verify prefix with,
    writing into the caller's output and lse when it passes them."""
    if kv_cache_format == "dsv4_fp8":
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            dpsk_v4_fp8_attention_fwd,
        )

        return dpsk_v4_fp8_attention_fwd(
            q,
            kv_cache,
            None,
            None,
            None,
            None,
            softmax_scale=sm_scale,
            indices=sparse_indices,
            attn_sink=sinks,
            extra_k_cache=extra_kv_cache,
            extra_indices_in_kvcache=extra_sparse_indices,
            topk_length=sparse_lens,
            extra_topk_length=extra_sparse_lens,
        )
    if block_table is None or cache_seqlens is None or causal or sinks is not None:
        raise ValueError("dense baseline: block_table and cache_seqlens only")
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    batch, query_length, n_heads, dim = q.shape
    if query_length != 1:
        raise ValueError("dense baseline: one query per row")
    if output is None:
        output = q.new_empty(batch, query_length, n_heads, head_dim_v)
    _, row_lse = mla_gluon(
        q[:, 0, :, :head_dim_v],
        q[:, 0, :, head_dim_v:],
        kv_cache.view(-1, dim),
        output.view(batch, n_heads, head_dim_v),
        block_table,
        cache_seqlens,
        sm_scale,
        kv_pe_offset=head_dim_v,
        use_2d_view=True,
        kv_scale=kv_scale,
        return_lse=True,
    )
    row_lse = row_lse.reshape(batch, query_length, n_heads)
    if lse is None:
        lse = row_lse
    else:
        lse.copy_(row_lse)
    return (output, lse) if return_lse else output


def fp8_paged_mqa_logits_baseline(
    q, kv_cache, weights, context_lens, block_tables, logits, *, page_size, preshuffled
):
    """aiter's kernel, with the max_model_len the route drops: the logits' width."""
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        logits,
        context_lens,
        block_tables,
        logits.shape[1],
        Preshuffle=preshuffled,
        KVBlockSize=page_size,
    )
    return logits
