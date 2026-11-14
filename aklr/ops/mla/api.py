# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.mla.contract import FLASH_MLA_WITH_KVCACHE, FP8_PAGED_MQA_LOGITS, MLA_DECODE_FWD

__all__ = ["flash_mla_with_kvcache", "fp8_paged_mqa_logits", "mla_decode_fwd"]


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
    rope_freqs=None,
    positions=None,
): ...


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


@api(MLA_DECODE_FWD)
def mla_decode_fwd(
    q,
    kv_buffer,
    kv_indptr,
    kv_indices,
    q_scale,
    kv_scale,
    sm_scale=None,
    o=None,
    lse=None,
): ...
