# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route, sglang_version
from aklr.plugins.sglang.layers.sparse_mla.contract import ROPE_DIM, SPARSE_MLA_VERIFY

__all__ = ["sparse_mla_verify", "triton_sparse_mla_verify"]


@api(SPARSE_MLA_VERIFY)
def sparse_mla_verify(q, kv_cache, topk_indices, sm_scale, native="tilelang"): ...


# The DSA backend's tilelang sparse MLA (dsa_decode_impl "tilelang"): target_verify,
# draft_extend and the draft decodes, a query per token over that token's own top-k slots
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_tilelang",
    versions=("0.5.22", "0.5.23"),
)
def _forward_tilelang(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale):
    """tilelang_sparse_fwd's attention as one op call, returned as _forward_tilelang returns it,
    [1, tokens, heads, v_head_dim] on 0.5.22, [tokens, heads, v_head_dim] on 0.5.23; a bf16 q
    becomes the pool's fp8 first, as tilelang does."""
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
    if isinstance(output, FallbackReason) or not _tilelang_keeps_batch_axis():
        return output
    return output.unsqueeze(0)


@functools.cache
def _tilelang_keeps_batch_axis() -> bool:
    """Whether _forward_tilelang returns the kernel's batch axis: 0.5.23 squeezes it."""
    return sglang_version() == "0.5.22"


# forward_extend's sparse MLA under the "triton" DSA backend, from the triton_sparse_mla_fwd
# route (layers/dsa_backend) on what its prefill op does not take: target_verify and
# draft_extend_v2 reach it with a query per token over that token's own top-k
def triton_sparse_mla_verify(q, kv, indices, sm_scale, max_topk_length):
    """triton_sparse_mla_fwd's attention as one op call on the whole q, returned as it
    returns it, [1, tokens, heads, d_v]. A table cut by a context shorter than its width stays
    on sglang, whose kernel then reads less of it."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend, has_forward_context

    if (
        (max_topk_length is not None and max_topk_length < indices.shape[-1])
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
