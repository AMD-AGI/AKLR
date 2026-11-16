# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend.api import _whole_query
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


# forward_extend's sparse MLA under the "triton" DSA backend, imported at each call:
# target_verify and draft_extend_v2 reach it with a query per token over that token's own top-k
@sglang_route(
    "sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd",
    versions=("0.5.22",),
)
def _triton_sparse_mla_fwd(
    q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None, max_topk_length=None
):
    """triton_sparse_mla_fwd's attention as one op call on the whole q, returned as it
    returns it, [1, tokens, heads, d_v]. A table cut by topk_length or by a context shorter
    than its width stays on sglang, whose kernel then reads less of it."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend, has_forward_context

    q = _whole_query(q_nope, q_rope)
    if (
        q is None
        or d_v != q_nope.shape[-1]
        or topk_length is not None
        or (max_topk_length is not None and max_topk_length < indices.shape[-1])
        or not has_forward_context()
        or not isinstance(get_attn_backend(), DeepseekSparseAttnBackend)
        or get_attn_backend().dcp_size > 1
    ):
        return FallbackReason.OUT_OF_CONTRACT
    output = sparse_mla_verify(
        q=q,
        kv_cache=kv.view(-1, kv.shape[-1]),
        topk_indices=indices.squeeze(1),
        sm_scale=sm_scale,
        native="triton",
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)
