# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23 runs on HIP by default: for DeepseekSparseAttnBackend._forward_triton_decode,
its triton split-K sparse MLA decode; for forward_extend's sparse MLA prefill, triton_sparse_mla_fwd;
for the DSA indexer's decode top-k, sgl_kernel's."""

import torch

from aklr.core.contract.baseline import OpBaseline
from aklr.plugins.sglang.layers.dsa_backend.reference import ROPE_DIM

__all__ = ["DsaSparseMlaDecodeBaseline", "DsaSparseMlaPrefillBaseline", "DsaTopkTransformBaseline"]


class DsaSparseMlaDecodeBaseline(OpBaseline):
    def baseline(self, *, q, kv_cache, topk_indices, sm_scale, output=None):
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
        return {"output": attended.squeeze(0)}


class DsaSparseMlaPrefillBaseline(OpBaseline):
    def baseline(self, *, q, kv_cache, topk_indices, sm_scale, output=None):
        """triton_sparse_mla_fwd as forward_extend calls it (sparse prefill: max_topk_length > 512)."""
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd

        latent = q.shape[-1] - ROPE_DIM
        attended = triton_sparse_mla_fwd(
            q_nope=q[..., :latent],
            q_rope=q[..., latent:],
            kv=kv_cache.view(kv_cache.shape[0], 1, -1),
            indices=topk_indices.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=latent,
        )
        # [1, tokens, heads, latent], as the backend returns it
        return {"output": attended.squeeze(0)}


class DsaTopkTransformBaseline(OpBaseline):
    def baseline(self, *, score, lengths, page_table_size_1, cu_seqlens_q, out_page_indices):
        """sgl_kernel's fast_topk_transform_fused, into the caller's out_page_indices."""
        import sgl_kernel  # noqa: F401, registers torch.ops.sgl_kernel

        torch.ops.sgl_kernel.fast_topk_transform_fused(
            score, lengths, out_page_indices, page_table_size_1, cu_seqlens_q, None
        )
        return {"out_page_indices": out_page_indices}
