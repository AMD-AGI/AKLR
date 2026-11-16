# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.sparse_mla.contract import ROPE_DIM, SPARSE_MLA_VERIFY

__all__ = ["sparse_mla_verify"]


@api(SPARSE_MLA_VERIFY)
def sparse_mla_verify(q, kv_cache, topk_indices, sm_scale, native="tilelang"): ...


# The DSA backend's tilelang sparse MLA (dsa_decode_impl "tilelang"): target_verify,
# draft_extend and the draft decodes, a query per token over that token's own top-k slots
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_tilelang",
    versions=("0.5.22",),
)
def _forward_tilelang(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale):
    """tilelang_sparse_fwd's attention as one op call, returned as it returns it,
    [1, tokens, heads, v_head_dim]; a bf16 q becomes the pool's fp8 first, as tilelang does."""
    import torch

    width = kv_cache.shape[-1]
    # a decode context parallel rank holds every dcp_size-th slot under global indices
    if (
        self.dcp_size > 1
        or kv_cache.dtype != torch.float8_e4m3fn
        or q_all.dim() != 3
        or q_all.shape[-1] != width
        or v_head_dim != width - ROPE_DIM
        or not q_all.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    output = sparse_mla_verify(
        q=q_all.to(kv_cache.dtype),
        kv_cache=kv_cache.view(-1, width),
        topk_indices=page_table_1,
        sm_scale=sm_scale,
        native="tilelang",
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)

