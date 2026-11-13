# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA decode vLLM's aiter MLA backend runs under decode context parallelism, which calls
aiter's own entry point for its LSE: AKLR's ``mla_decode_fwd`` op."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import aiter as aklr_aiter
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLAImpl._forward_mqa",
    "aiter.mla.mla_decode_fwd",
    versions=("0.33.0rc1.dev910",),
)
def _mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size=1,
    nhead_kv=1,
    sm_scale=None,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
    intra_batch_mode=False,
    return_logits=False,
    return_lse=False,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    causal=True,
    kv_layout="token_major",
    seq_lens=None,
    page_table=None,
    kv_csr_layout="page",
    fast_mode=True,
):
    # aiter 0.1.28's other KV layouts and its paged CSR metadata are not the contract's pool
    if (
        kv_layout != "token_major"
        or kv_csr_layout != "page"
        or seq_lens is not None
        or page_table is not None
    ):
        return FallbackReason.OUT_OF_CONTRACT
    return aklr_aiter.mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        page_size=page_size,
        nhead_kv=nhead_kv,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        num_kv_splits_indptr=num_kv_splits_indptr,
        q_scale=q_scale,
        kv_scale=kv_scale,
        return_logits=return_logits,
        return_lse=return_lse,
        g_kv_indptr=g_kv_indptr,
        cp_world_size=cp_world_size,
    )
