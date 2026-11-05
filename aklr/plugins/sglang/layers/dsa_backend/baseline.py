# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs for the DSA attention on HIP by default (the triton backend): for
DeepseekSparseAttnBackend._forward_triton_decode, its triton split-K sparse MLA decode; for the
prefill, the triton sparse MLA that forward_extend calls."""

from __future__ import annotations

from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM

__all__ = ["dsa_sparse_mla_decode_baseline", "dsa_sparse_mla_prefill_baseline"]


def dsa_sparse_mla_decode_baseline(q, kv_cache, topk_indices, sm_scale, output=None):
    """triton_sparse_mla_decode_splitk as _forward_triton_decode calls it: q as its latent
    and RoPE views, the pool as [rows, 1, kv_dim], the top-k as [tokens, 1, topk]; its
    split-K workspace grows per stream and is kept (the backend's is too)."""
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )

    latent = q.shape[-1] - ROPE_DIM
    attended = triton_sparse_mla_decode_splitk(
        q_nope=q[..., :latent],
        q_rope=q[..., latent:],
        kv=kv_cache.view(kv_cache.shape[0], 1, -1),
        indices=topk_indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=latent,
    )
    # [1, tokens, heads, latent], as the backend returns it
    return attended.squeeze(0)


def dsa_sparse_mla_prefill_baseline(q, kv_cache, topk_indices, sm_scale, output=None):
    """triton_sparse_mla_fwd as forward_extend calls it: q as its latent and RoPE views, the
    pool as [rows, 1, kv_dim], the top-k as [tokens, 1, topk], no per-row top-k length, and
    max_topk_length the top-k width. That is min(max_seq_len_k, topk) there: a sparse
    prefill's longest sequence holds more than topk tokens, as shorter ones take the dense
    MHA path."""
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd

    latent = q.shape[-1] - ROPE_DIM
    attended = triton_sparse_mla_fwd(
        q_nope=q[..., :latent],
        q_rope=q[..., latent:],
        kv=kv_cache.view(kv_cache.shape[0], 1, -1),
        indices=topk_indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=latent,
        topk_length=None,
        max_topk_length=topk_indices.shape[-1],
    )
    # [1, tokens, heads, latent], as the kernel returns it
    return attended.squeeze(0)
