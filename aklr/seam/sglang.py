###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.ops.mla import flash_mla_with_kvcache
from aklr.ops.topk import topk_transform_paged
from aklr.seam import adapter


@adapter(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel.dpsk_v4_fp8_attention_fwd",
    flash_mla_with_kvcache,
)
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
    # block_table, cache_seqlens, head_dim_v, scheduling, causal and is_fp8_kvcache reach no schema argument
    return {
        "q": q,
        "kv_cache": k_cache,
        "sparse_indices": indices,
        "sparse_lens": topk_length,
        "extra_kv_cache": extra_k_cache,
        "extra_sparse_indices": extra_indices_in_kvcache,
        "extra_sparse_lens": extra_topk_length,
        "sm_scale": softmax_scale,
        "sinks": attn_sink,
        "kv_cache_format": "dsv4_fp8",
    }


@adapter(
    "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2",
    topk_transform_paged,
)
def _(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    metadata,
    out_raw_indices=None,
):
    # metadata is sglang's launch plan, not the schema's
    if out_raw_indices is not None:
        return None
    return {
        "scores": scores,
        "seq_lens": seq_lens,
        "page_tables": page_tables,
        "k": out_page_indices.shape[-1],
        "out_page_indices": out_page_indices,
        "page_size": page_size,
    }
