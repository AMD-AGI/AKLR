# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 and 0.5.23 run on HIP for a speculative step's DSA sparse MLA: native
"tilelang", DeepseekSparseAttnBackend._forward_tilelang's tilelang_sparse_fwd; native "triton",
forward_extend's triton_sparse_mla_fwd."""

from __future__ import annotations

from aklr.plugins.sglang.layers.sparse_mla.contract import ROPE_DIM

__all__ = ["sparse_mla_verify_baseline"]


def sparse_mla_verify_baseline(q, kv_cache, topk_indices, sm_scale, native="tilelang", output=None):
    """The native kernel as the backend calls it, the pool as [rows, 1, kv_dim] and the top-k
    as [tokens, 1, topk]; tilelang's top-k padded with -1 to 64 entries a block, as
    _forward_tilelang pads it."""
    import torch

    latent = q.shape[-1] - ROPE_DIM
    kv = kv_cache.view(kv_cache.shape[0], 1, -1)
    if native == "tilelang":
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        padding = (-topk_indices.shape[-1]) % 64
        if padding:
            topk_indices = torch.nn.functional.pad(topk_indices, (0, padding), value=-1)
        attended = tilelang_sparse_fwd(
            q=q, kv=kv, indices=topk_indices.unsqueeze(1), sm_scale=sm_scale, d_v=latent
        )
    else:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd

        # forward_extend's max_topk_length: the table's width once a context fills it
        attended = triton_sparse_mla_fwd(
            q_nope=q[..., :latent],
            q_rope=q[..., latent:],
            kv=kv,
            indices=topk_indices.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=latent,
            topk_length=None,
            max_topk_length=topk_indices.shape[-1],
        )
    # [1, tokens, heads, latent], as the backend returns it
    return attended.squeeze(0)
