###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.attention.schema import paged_attention_decode_op_schema, unified_attention_op_schema

__all__ = ["paged_attention_decode", "unified_attention"]


@aklr_api(paged_attention_decode_op_schema)
def paged_attention_decode(
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
    out=None,
): ...


@aklr_api(unified_attention_op_schema)
def unified_attention(
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
    out=None,
): ...
