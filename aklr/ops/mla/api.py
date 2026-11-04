###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.mla.schema import (
    flash_mla_with_kvcache_op_schema,
    fp8_paged_mqa_logits_op_schema,
)

__all__ = ["flash_mla_with_kvcache", "fp8_paged_mqa_logits"]


@aklr_api(flash_mla_with_kvcache_op_schema)
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
): ...


@aklr_api(fp8_paged_mqa_logits_op_schema)
def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    page_size,
    logits=None,
    *,
    preshuffled=True,
): ...
