###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the attention op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import torch


def unified_attention_baseline(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    max_seqlen_q,
    window_size=(-1, -1),
    causal=True,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out=None,
    **_,
):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    if out is None:
        out = torch.empty_like(q)
    # max_seqlen_k: the block table's width, what vLLM passes when it captures a CUDA graph
    return lambda: unified_attention(
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


def paged_attention_v1_baseline(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    scale,
    sliding_window=0,
    out=None,
    **_,
):
    from aiter.ops.attention import paged_attention_v1

    if out is None:
        out = torch.empty_like(query)
    batch, n_heads, head_dim = query.shape
    max_context_len = block_tables.shape[1] * key_cache.shape[1]
    # per partition of 256 tokens, aiter's default: head_dim partial outputs in query's dtype, two fp32 statistics
    partitions = -(-max_context_len // 256)
    workspace = torch.empty(
        batch * n_heads * partitions * (head_dim * query.element_size() + 8),
        dtype=torch.uint8,
        device=query.device,
    )
    unit = torch.ones(1, dtype=torch.float32, device=query.device)
    # as vLLM's ROCM_AITER_FA backend calls it: unit kv scales, the block table's width as max_context_len
    return lambda: paged_attention_v1(
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
        None,
        "auto",
        "NHD",
        0.0,
        unit,
        unit,
        sliding_window=sliding_window,
    )
