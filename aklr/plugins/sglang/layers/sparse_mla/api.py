# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route, sglang_version
from aklr.plugins.sglang.layers.dsa_backend.api import dcp_size, dsa_sparse_mla_decode, dsa_sparse_mla_prefill
from aklr.plugins.sglang.layers.sparse_mla.contract import ROPE_DIM, SPARSE_MLA_VERIFY

__all__ = ["sparse_mla_verify", "triton_sparse_mla_verify"]


@api(SPARSE_MLA_VERIFY)
def sparse_mla_verify(q, kv_cache, topk_indices, sm_scale, native="tilelang"): ...


# dsa_sparse_mla_prefill's tilelang-native Points are every this many tokens: an extend in between
# runs as the next one, its added tokens reading no slot (-1 rows cost the kernel nothing)
_PREFILL_POINT_STEP = 1024


# The DSA backend's tilelang sparse MLA (dsa_decode_impl / dsa_prefill_impl "tilelang"): without
# speculative decoding a decode goes to dsa_sparse_mla_decode and an extend (new sequences, or onto
# a prefix) to dsa_sparse_mla_prefill, each against tilelang; speculative decoding's steps
# (target_verify, draft_extend, the draft decodes) to sparse_mla_verify
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_tilelang",
    # from the image's daily build of 0.5.23 on
    versions=("0.5.22", ">=0.5.23.dev20261120"),
)
def _forward_tilelang(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale):
    """tilelang_sparse_fwd's attention as one op call, returned as _forward_tilelang returns it,
    [1, tokens, heads, v_head_dim] on 0.5.22, [tokens, heads, v_head_dim] on 0.5.23; a bf16 q
    becomes the pool's fp8 first, as tilelang does."""
    import torch

    width = kv_cache.shape[-1]
    # a decode context parallel rank holds every dcp_size-th slot under global indices
    if (
        dcp_size(self) > 1
        or kv_cache.dtype != torch.float8_e4m3fn
        or q_all.dim() != 3
        or q_all.shape[-1] != width
        or v_head_dim != width - ROPE_DIM
        or not q_all.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    tokens = q_all.shape[0]
    phase = _tilelang_phase(tokens)
    q, table = q_all.to(kv_cache.dtype), page_table_1
    if phase == "prefill" and tokens % _PREFILL_POINT_STEP:
        q, table = _padded_to_a_point(q, table)
    op = {"decode": dsa_sparse_mla_decode, "prefill": dsa_sparse_mla_prefill}.get(phase, sparse_mla_verify)
    output = op(
        q=q,
        kv_cache=kv_cache.view(-1, width),
        topk_indices=table,
        sm_scale=sm_scale,
        native="tilelang",
    )
    if isinstance(output, FallbackReason):
        return output
    output = output[:tokens]
    return output.unsqueeze(0) if _tilelang_keeps_batch_axis() else output


def _padded_to_a_point(q, table):
    """q and the top-k table grown to the next multiple of _PREFILL_POINT_STEP tokens, the added
    tokens' rows all -1 (a token that reads no row: 0 out)."""
    tokens = q.shape[0]
    padded = -(-tokens // _PREFILL_POINT_STEP) * _PREFILL_POINT_STEP
    q_padded = q.new_zeros((padded, *q.shape[1:]))
    q_padded[:tokens] = q
    table_padded = table.new_full((padded, table.shape[-1]), -1)
    table_padded[:tokens] = table
    return q_padded, table_padded


def _tilelang_phase(tokens):
    """ "decode": a query per sequence; "prefill": an extend's tokens, new sequences' or onto a prefix
    (outside a graph capture); None: speculative decoding's steps (from the backend's host lengths,
    no device sync)."""
    import torch
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend, has_forward_context
    from sglang.srt.runtime_context import get_spec

    if get_spec().speculative_algorithm is not None or not has_forward_context():
        return None
    backend = get_attn_backend()
    if not isinstance(backend, DeepseekSparseAttnBackend):
        return None
    metadata = backend.forward_metadata
    extend_lens = metadata.dsa_extend_seq_lens_list
    if len(extend_lens) == tokens and all(n == 1 for n in extend_lens):
        return "decode"
    if not torch.cuda.is_current_stream_capturing() and sum(extend_lens) == tokens:
        return "prefill"
    return None


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
        or dcp_size(get_attn_backend()) > 1
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
