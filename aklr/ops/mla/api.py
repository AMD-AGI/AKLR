# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route, unused
from aklr.ops.mla.contract import FLASH_MLA_WITH_KVCACHE, FP8_PAGED_MQA_LOGITS

__all__ = ["flash_mla_with_kvcache", "fp8_paged_mqa_logits"]


@api(FLASH_MLA_WITH_KVCACHE)
def flash_mla_with_kvcache(
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
): ...


@aklr_route("sglang.kernels.ops.attention.dsa.tilelang_kernel.dpsk_v4_fp8_attention_fwd")
def _(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=False,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
):
    unused(block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits)
    unused(causal, is_fp8_kvcache)
    return flash_mla_with_kvcache(
        q=q,
        kv_cache=k_cache,
        sparse_indices=indices,
        sparse_lens=topk_length,
        extra_kv_cache=extra_k_cache,
        extra_sparse_indices=extra_indices_in_kvcache,
        extra_sparse_lens=extra_topk_length,
        sm_scale=softmax_scale,
        sinks=attn_sink,
        kv_cache_format="dsv4_fp8",
        return_lse=True,
        kv_cache_layout="padded_pages",
    )


@api(FP8_PAGED_MQA_LOGITS)
def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    logits,
    *,
    page_size,
    preshuffled=True,
): ...


@aklr_route("aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits")
def _(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    Preshuffle=False,
    KVBlockSize=1,
    ChunkK=256,
    TotalCuCount=None,
    WavePerEU=2,
    VarCtxSchedule=None,
):
    # logits' width carries max_model_len; the rest is tiling
    unused(max_model_len, ChunkK, TotalCuCount, WavePerEU, VarCtxSchedule)
    fp8_paged_mqa_logits(
        q=q_fp8,
        kv_cache=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_tables=kv_indices,
        logits=out_logits,
        page_size=KVBlockSize,
        preshuffled=Preshuffle,
    )
