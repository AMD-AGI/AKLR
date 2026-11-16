# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.24.dev809 runs on gfx950 for forward_triton_dcp_partial past its top-k
localization: sparse_mla_decode_gfx950 (SGLANG_ENABLE_ROCM_DSA_OPUS_DECODE, on by default), which
quantizes q's NoPE part to MXFP8 for its HIP kernel and merges its top-k splits in triton."""

from __future__ import annotations

__all__ = ["dsa_sparse_mla_dcp_partial_baseline"]

# Kimi-K3.1-Pro's DCP group: sglang's expected_topk is the global top-k over it, which only picks the
# kernel's splits
_DCP_SIZE = 8


def dsa_sparse_mla_dcp_partial_baseline(
    q_nope, q_rope, kv_cache, indices, topk_length, sm_scale=192**-0.5, output=None, lse=None
):
    """sparse_mla_decode_gfx950 as forward_triton_dcp_partial calls it: the pool as [slots, 1, 656],
    the indices as [rows, 1, topk], expected_topk the top-k's share of one rank; its output squeezed
    as the backend returns it."""
    from sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 import sparse_mla_decode_gfx950

    attended, row_lse = sparse_mla_decode_gfx950(
        q_nope=q_nope,
        q_rope=q_rope,
        kv=kv_cache.unsqueeze(1),
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        topk_length=topk_length,
        expected_topk=-(-indices.shape[-1] // _DCP_SIZE),
    )
    return attended.squeeze(0), row_lse
