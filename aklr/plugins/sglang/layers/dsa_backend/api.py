# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend.contract import DSA_SPARSE_MLA_DECODE, DSA_SPARSE_MLA_PREFILL

__all__ = ["dsa_sparse_mla_decode", "dsa_sparse_mla_prefill"]

# sglang's short-sequence path in triton_sparse_mla_fwd
_SGLANG_SHORT_TOPK = 512


@api(DSA_SPARSE_MLA_DECODE)
def dsa_sparse_mla_decode(q, kv_cache, topk_indices, sm_scale): ...


@api(DSA_SPARSE_MLA_PREFILL)
def dsa_sparse_mla_prefill(q, kv_cache, topk_indices, sm_scale): ...


# _forward_triton_decode, not forward_decode: sglang still picks the DSA decode
# implementation (triton, HIP's default), and this is the one the baseline measures.
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_triton_decode",
    versions=("0.5.22",),
)
def _forward_triton_decode(self, q_nope, q_rope, kv_cache, v_head_dim, page_table_1, sm_scale):
    """The decode's attention as one op call on the whole q, returned as sglang's kernel returns
    it, [1, tokens, heads, v_head_dim]; v_head_dim is q_nope's width, as sglang's kernel asserts."""
    from sglang.srt.runtime_context import get_spec

    q = _whole_query(q_nope, q_rope)
    # speculative decoding's draft steps stay on sglang
    if q is None or get_spec().speculative_algorithm is not None:
        return FallbackReason.OUT_OF_CONTRACT
    output = dsa_sparse_mla_decode(
        q=q,
        kv_cache=kv_cache.view(-1, kv_cache.shape[-1]),
        topk_indices=page_table_1,
        sm_scale=sm_scale,
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)


# HIP's DSA prefill; forward_extend imports it at each call.
@sglang_route(
    "sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd",
    versions=("0.5.22",),
)
def _triton_sparse_mla_fwd(
    q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None, max_topk_length=None
):
    """New sequences' prefill as one op call; tokens after a cached prefix stay on sglang."""
    import torch
    from sglang.srt.runtime_context import get_spec

    q = _whole_query(q_nope, q_rope)
    if (
        q is None
        or d_v != q_nope.shape[-1]
        or get_spec().speculative_algorithm is not None
        or topk_length is not None
        or (max_topk_length is not None and max_topk_length <= _SGLANG_SHORT_TOPK)
        or torch.cuda.is_current_stream_capturing()
        or not _is_new_sequences(q.shape[0])
    ):
        return FallbackReason.OUT_OF_CONTRACT
    output = dsa_sparse_mla_prefill(
        q=q,
        kv_cache=kv.view(-1, kv.shape[-1]),
        topk_indices=indices.squeeze(1),
        sm_scale=sm_scale,
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)


def _is_new_sequences(tokens):
    """Whether this forward's sequences have no cached prefix (host lengths, no device sync)."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend, has_forward_context

    if not has_forward_context():
        return False
    backend = get_attn_backend()
    if not isinstance(backend, DeepseekSparseAttnBackend):
        return False
    metadata = backend.forward_metadata
    return metadata.seq_lens_sum == sum(metadata.dsa_extend_seq_lens_list) == tokens


def _whole_query(q_nope, q_rope):
    """The [tokens, heads, latent + rope] q whose two views forward_decode passes; None for two
    tensors apart."""
    tokens, heads, latent = q_nope.shape
    width = latent + q_rope.shape[-1]
    if not (
        q_rope.dtype == q_nope.dtype
        and q_rope.shape[:2] == q_nope.shape[:2]
        and q_nope.stride() == q_rope.stride() == (heads * width, width, 1)
        and q_rope.untyped_storage().data_ptr() == q_nope.untyped_storage().data_ptr()
        and q_rope.storage_offset() == q_nope.storage_offset() + latent
    ):
        return None
    return q_nope.as_strided((tokens, heads, width), q_nope.stride())
