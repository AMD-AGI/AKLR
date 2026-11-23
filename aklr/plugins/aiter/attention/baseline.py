# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["paged_attention_decode_baseline", "unified_attention_baseline"]

_PARTITION_SIZE = 256  # aiter's paged_attention_v1 default


def paged_attention_decode_baseline(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    scale,
    *,
    kv_cache_dtype="auto",
    kv_cache_layout="NHD",
    kv_cache_storage="row_major",
    alibi_slopes=None,
    sliding_window=0,
    logits_soft_cap=0.0,
    out_dtype=None,
    out=None,
):
    """aiter's paged_attention_v1 as vLLM's ROCM_AITER_FA backend calls it, with the
    workspace it sizes, the block table's width as max_context_len and unit kv scales.
    The kernel reads either cache storage by its strides."""
    import torch
    from aiter.ops.attention import paged_attention_v1

    if out is None:
        out = torch.empty_like(query)
    batch, n_heads, head_dim = query.shape
    max_context_len = block_tables.shape[1] * key_cache.shape[1]
    partitions = -(-max_context_len // _PARTITION_SIZE)
    # per partition: head_dim partial outputs in query's dtype, two fp32 statistics
    workspace = torch.empty(
        batch * n_heads * partitions * (head_dim * query.element_size() + 8),
        dtype=torch.uint8,
        device=query.device,
    )
    unit = torch.ones(1, dtype=torch.float32, device=query.device)
    return paged_attention_v1(
        out,
        workspace,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        None,
        context_lens,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        unit,
        unit,
        sliding_window=sliding_window,
    )


def unified_attention_baseline(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    *,
    max_seqlen_q,
    window_size=(-1, -1),
    causal=True,
    kv_cache_storage="row_major",
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out_dtype=None,
    out=None,
):
    """aiter's Triton unified_attention as vLLM's ROCM_AITER_FA backend calls it, with the
    block table's width as max_seqlen_k, what vLLM passes when it captures a CUDA graph
    (its max_model_len). The kernel reads either cache storage by its strides."""
    import torch
    from aiter.ops.triton.attention.unified_attention import unified_attention

    if out is None:
        out = torch.empty_like(q)
    unified_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        block_table.shape[1] * k.shape[1],
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
    )
    return out
