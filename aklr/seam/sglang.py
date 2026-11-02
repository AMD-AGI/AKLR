###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import importlib

from aklr.ops.mla import flash_mla_with_kvcache
from aklr.ops.topk import topk_transform_paged
from aklr.seam import adapter


def _dpsk_v4_fp8_attention_baseline(
    original,
    q,
    kv_cache,
    sparse_indices=None,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    sm_scale=None,
    sinks=None,
    kv_cache_format="dsv4_fp8",
    block_table=None,
    cache_seqlens=None,
    causal=False,
    kv_scale=1.0,
    output=None,
    lse=None,
    **_,
):
    if kv_cache_format != "dsv4_fp8":
        # dense: aiter's Gluon MLA decode, the kernel sglang serves Kimi K3's DSPARK verify prefix with
        if block_table is None or cache_seqlens is None or causal or sinks is not None or q.shape[1] != 1:
            return None
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        batch, _, n_heads, dim = q.shape
        v_dim = 512
        out = q.new_empty(batch, 1, n_heads, v_dim) if output is None else output

        def dense():
            _, row_lse = mla_gluon(
                q[:, 0, :, :v_dim],
                q[:, 0, :, v_dim:],
                kv_cache.view(-1, dim),
                out.view(batch, n_heads, v_dim),
                block_table,
                cache_seqlens,
                q.shape[-1] ** -0.5 if sm_scale is None else sm_scale,
                kv_pe_offset=v_dim,
                use_2d_view=True,
                kv_scale=kv_scale,
                return_lse=True,
            )
            if lse is not None:
                lse.copy_(row_lse.reshape(batch, 1, n_heads))

        return dense
    return lambda: original(
        q=q,
        k_cache=kv_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=None,
        tile_scheduler_metadata=None,
        softmax_scale=sm_scale,
        indices=sparse_indices,
        attn_sink=sinks,
        extra_k_cache=extra_kv_cache,
        extra_indices_in_kvcache=extra_sparse_indices,
        topk_length=sparse_lens,
        extra_topk_length=extra_sparse_lens,
    )


@adapter(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel.dpsk_v4_fp8_attention_fwd",
    flash_mla_with_kvcache,
    _dpsk_v4_fp8_attention_baseline,
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


def _topk_transform_baseline(original, scores, seq_lens, page_tables, out_page_indices, page_size=1, **_):
    metadata = importlib.import_module(original.__module__).plan_topk_v2(seq_lens)
    return lambda: original(
        scores=scores,
        seq_lens=seq_lens,
        page_tables=page_tables,
        out_page_indices=out_page_indices,
        page_size=page_size,
        metadata=metadata,
    )


@adapter(
    "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2",
    topk_transform_paged,
    _topk_transform_baseline,
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
