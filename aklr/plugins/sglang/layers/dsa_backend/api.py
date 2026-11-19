# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.ops.mla import fp8_paged_mqa_logits, fp8_paged_mqa_logits_verify
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend.contract import (
    DSA_SPARSE_MLA_DECODE,
    DSA_SPARSE_MLA_PREFILL,
    DSA_TOPK_TRANSFORM,
)

__all__ = ["dsa_sparse_mla_decode", "dsa_sparse_mla_prefill", "dsa_topk_transform"]

# sglang's short-sequence path in triton_sparse_mla_fwd
_SGLANG_SHORT_TOPK = 512


@api(DSA_SPARSE_MLA_DECODE)
def dsa_sparse_mla_decode(q, kv_cache, topk_indices, sm_scale): ...


@api(DSA_SPARSE_MLA_PREFILL)
def dsa_sparse_mla_prefill(q, kv_cache, topk_indices, sm_scale): ...


@api(DSA_TOPK_TRANSFORM)
def dsa_topk_transform(score, lengths, page_table_size_1, cu_seqlens_q, out_page_indices): ...


# _forward_triton_decode, not forward_decode: sglang still picks the DSA decode
# implementation (triton, HIP's default), and this is the one the baseline measures.
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_triton_decode",
    versions=("0.5.23",),
)
def _forward_triton_decode(self, q_nope, q_rope, kv_cache, v_head_dim, page_table_1, sm_scale):
    """The decode's attention as one op call on the whole q, returned as sglang returns it,
    [tokens, heads, v_head_dim]; v_head_dim is q_nope's width, as sglang's kernel asserts."""
    q = _whole_query(q_nope, q_rope)
    # speculative decoding's draft steps are decodes too, a query per sequence over its own
    # top-k; a decode context parallel rank holds every dcp_size-th slot under global indices
    if q is None or self.dcp_size > 1:
        return FallbackReason.OUT_OF_CONTRACT
    return dsa_sparse_mla_decode(
        q=q,
        kv_cache=kv_cache.view(-1, kv_cache.shape[-1]),
        topk_indices=page_table_1,
        sm_scale=sm_scale,
    )


# (query rows, whether a target verify) of the paged top-k sglang's indexer is running, noted by the route on
# Indexer._get_topk_paged right before it asks for the logits and taken by that logits call: under DCP the
# verify rows a request reach the logits as rows of their own, each with the request's table row, as a decode
# batch's rows do, so only the forward mode tells the two apart
_PAGED_TOPK_MODE: list[tuple[int, bool] | None] = [None]


@sglang_route(
    "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged",
    versions=("0.5.24.dev809",),
)
def _get_topk_paged(self, forward_batch, layer_id, q_fp8, *args, **kwargs):
    """No op: notes whether this paged top-k is a target verify for _verify_logits; sglang runs its own."""
    _PAGED_TOPK_MODE[0] = (q_fp8.shape[0], forward_batch.forward_mode.is_target_verify())
    return FallbackReason.OUT_OF_CONTRACT


@sglang_route(
    "sglang.kernels.ops.attention.dsa.paged_mqa_logits.aiter_paged_mqa_logits",
    versions=("0.5.22", "0.5.23", "0.5.24.dev809"),
)
def _aiter_paged_mqa_logits(
    q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len, *, preshuffle, kv_block_size
):
    """The DSA indexer's logits as one op call, into the [rows, max_seq_len] float32 tensor
    sglang's wrapper allocates; verify rows fp8_paged_mqa_logits_verify takes go to it."""
    import torch

    logits = _verify_logits(
        q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len, preshuffle, kv_block_size
    )
    if logits is not None:
        return logits
    return fp8_paged_mqa_logits(
        q=q_fp8.unsqueeze(1),
        kv_cache=kv_cache_fp8.view(torch.uint8),
        weights=weights,
        context_lens=seq_lens,
        block_tables=block_tables,
        logits=q_fp8.new_empty((q_fp8.shape[0], max_seq_len), dtype=torch.float32),
        page_size=kv_block_size,
        preshuffled=preshuffle,
    )


# HIP's DSA top-k: sglang turns the v2 paged top-k off there for DSA models
@sglang_route("sgl_kernel.top_k.fast_topk_transform_fused", versions=("0.5.22", "0.5.23"))
def _fast_topk_transform_fused(score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts=None):
    """The DSA indexer's paged top-k as one op call, into the [rows, topk] int32 tensor
    sgl_kernel allocates; a ragged row_starts stays on sgl_kernel."""
    import torch

    if row_starts is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return dsa_topk_transform(
        score=score,
        lengths=lengths,
        page_table_size_1=page_table_size_1,
        cu_seqlens_q=cu_seqlens_q,
        out_page_indices=score.new_empty((score.shape[0], topk), dtype=torch.int32),
    )


# HIP's DSA prefill, and with speculative decoding its target_verify and draft_extend_v2;
# forward_extend imports it at each call.
@sglang_route(
    "sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd",
    versions=("0.5.22", "0.5.23"),
)
def _triton_sparse_mla_fwd(
    q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None, max_topk_length=None
):
    """New sequences' prefill as one op call; any other call as sparse_mla_verify's, a query per
    token over that token's own top-k."""
    import torch
    from sglang.srt.runtime_context import get_spec

    q = _whole_query(q_nope, q_rope)
    if q is None or d_v != q_nope.shape[-1] or topk_length is not None:
        return FallbackReason.OUT_OF_CONTRACT
    if (
        get_spec().speculative_algorithm is None
        and (max_topk_length is None or max_topk_length > _SGLANG_SHORT_TOPK)
        and not torch.cuda.is_current_stream_capturing()
        and _is_new_sequences(q.shape[0])
    ):
        output = dsa_sparse_mla_prefill(
            q=q,
            kv_cache=kv.view(-1, kv.shape[-1]),
            topk_indices=indices.squeeze(1),
            sm_scale=sm_scale,
        )
        if not isinstance(output, FallbackReason):
            return output.unsqueeze(0)
    from aklr.plugins.sglang.layers.sparse_mla.api import triton_sparse_mla_verify

    return triton_sparse_mla_verify(q, kv, indices, sm_scale, max_topk_length)


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


def _verify_logits(
    q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len, preshuffle, kv_block_size
):
    """The [rows, max_seq_len] logits of fp8_paged_mqa_logits_verify, else None. Under decode context
    parallelism sglang's indexer runs decode and verify alone, and with speculative decoding the target's
    decode is a verify: speculative_num_draft_tokens rows a request, each passed the request's block table
    row (repeat_interleave). Only rows the indexer's paged top-k noted as a target verify's are regrouped: a
    decode batch (a draft's, or any batch whose row count the draft tokens divide) stays on the row op."""
    import torch

    mode, _PAGED_TOPK_MODE[0] = _PAGED_TOPK_MODE[0], None
    if mode != (q_fp8.shape[0], True):
        return None
    try:
        from sglang.srt.runtime_context import get_parallel, get_spec
    except ImportError:
        return None
    if get_parallel().attn_dcp_size <= 1:
        return None
    rows, next_n = q_fp8.shape[0], get_spec().speculative_num_draft_tokens or 0
    if next_n < 2 or rows == 0 or rows % next_n:
        return None
    # the rows are regrouped by views: a strided caller stays on the row op
    if not (q_fp8.is_contiguous() and weights.is_contiguous() and seq_lens.is_contiguous()):
        return None
    batch = rows // next_n
    args = {
        "q": q_fp8.view(batch, next_n, *q_fp8.shape[1:]),
        "kv_cache": kv_cache_fp8.view(torch.uint8),
        "weights": weights.view(batch, next_n, -1),
        "context_lens": seq_lens.view(batch, next_n),
        "logits": q_fp8.new_empty((batch, next_n, max_seq_len), dtype=torch.float32),
        "page_size": kv_block_size,
        "preshuffled": preshuffle,
    }
    # each request's first row; table rows past the queries' are not read, as aiter does not
    tables = block_tables[:rows:next_n]
    # asked with a contiguous stand-in: the rows are copied only for a call the op takes
    stand_in = torch.empty(tables.shape, dtype=tables.dtype, device=tables.device)
    reason = fp8_paged_mqa_logits_verify.fallback_reason(block_tables=stand_in, **args)
    del stand_in
    if reason is not None:
        return None
    logits = fp8_paged_mqa_logits_verify(block_tables=tables.contiguous(), **args)
    return None if isinstance(logits, FallbackReason) else logits.view(rows, max_seq_len)
