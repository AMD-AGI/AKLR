###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the mla op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import torch


def deepgemm_fp8_paged_mqa_logits_baseline(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    page_size,
    logits=None,
    preshuffled=True,
    **_,
):
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    if logits is None:
        return None
    return lambda: deepgemm_fp8_paged_mqa_logits(
        q_fp8=q,
        kv_cache=kv_cache,
        weights=weights,
        out_logits=logits,
        context_lens=context_lens,
        kv_indices=block_tables,
        max_model_len=logits.shape[1],
        Preshuffle=preshuffled,
        KVBlockSize=page_size,
    )


def dpsk_v4_fp8_attention_baseline(
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
    cu_seqlens_k=None,
    causal=False,
    kv_scale=1.0,
    output=None,
    lse=None,
    **_,
):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import dpsk_v4_fp8_attention_fwd

    if kv_cache_format != "dsv4_fp8":
        # dense: aiter's Gluon MLA decode, the kernel sglang serves Kimi K3's DSPARK verify prefix with, over the
        # 2-D block table read up to cache_seqlens or the flat one cut by cu_seqlens_k
        lengths = cache_seqlens if cu_seqlens_k is None else cu_seqlens_k
        if block_table is None or lengths is None or causal or sinks is not None or q.shape[1] != 1:
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
                lengths,
                q.shape[-1] ** -0.5 if sm_scale is None else sm_scale,
                kv_pe_offset=v_dim,
                use_2d_view=cu_seqlens_k is None,
                kv_scale=kv_scale,
                return_lse=True,
            )
            if lse is not None:
                lse.copy_(row_lse.reshape(batch, 1, n_heads))

        return dense
    return lambda: dpsk_v4_fp8_attention_fwd(
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


def mla_decode_fwd_baseline(
    q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=None, o=None, lse=None, **_
):
    """aiter's ASM MLA decode as sglang calls it under decode context parallelism: fp8 q and pool, over its
    persistent schedule (fast mode, 64 splits a request), planned here as a framework plans it once a step."""
    from aiter.mla import mla_decode_fwd
    from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

    if o is None or sm_scale is None:
        return None
    batch, n_heads = q.shape[:2]
    n_splits = 64  # sglang's decode without speculation
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    last_page_lens = torch.ones(batch, dtype=torch.int32, device=q.device)
    sizes = get_mla_metadata_info_v1(
        batch,
        1,
        n_heads,
        q.dtype,
        kv_buffer.dtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=n_splits,
        intra_batch_mode=False,
    )
    work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = (
        torch.empty(size, dtype=dtype, device=q.device) for size, dtype in sizes
    )
    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last_page_lens,
        n_heads,
        1,
        False,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=n_splits,
        intra_batch_mode=False,
        dtype_q=q.dtype,
        dtype_kv=kv_buffer.dtype,
    )

    def decode():
        _, final_lse = mla_decode_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_lens,
            1,
            sm_scale=sm_scale,
            num_kv_splits=n_splits,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=q_scale,
            kv_scale=kv_scale,
            intra_batch_mode=False,
            return_lse=True,
        )
        if lse is not None:
            lse.copy_(final_lse.view(batch, n_heads))

    return decode
