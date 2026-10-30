# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs for DeepseekSparseAttnBackend._forward_triton_decode, the DSA
decode on HIP by default: its triton split-K sparse MLA decode."""

from __future__ import annotations

from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM

__all__ = ["dsa_sparse_mla_decode_baseline"]


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
