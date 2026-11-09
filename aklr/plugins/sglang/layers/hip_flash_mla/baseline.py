# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["deepseek_v4_sparse_mla_decode_inv_rope_baseline"]


def deepseek_v4_sparse_mla_decode_inv_rope_baseline(
    q,
    kv_cache,
    sparse_indices,
    sinks,
    rope_freqs,
    positions,
    kv_cache_layout="row_major",
    extra_kv_cache=None,
    extra_sparse_indices=None,
    sm_scale=0.04419417382415922,
    output=None,
):
    """What sglang 0.5.23's aiter_sparse_decode_fwd runs for DeepSeek-V4's fp8 pools with
    inv_rope outside deterministic inference: aiter's gfx950 pa_decode_sparse (its own split
    count below 1024 rows, one split from 1024), then on one split sglang's flat-row inverse
    RoPE in place over the output's last 64 lanes, or on several sglang's split-K combine with
    the rotation folded in; here writing the caller's output instead of a new tensor."""
    import torch
    import triton
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
    from sglang.kernels.ops.attention.dsv4.rope import apply_rotary_emb_flat_kernel
    from sglang.kernels.ops.attention.sparse_decode_reduce_hip import _gfx_sparse_decode_reduce_kernel
    from sglang.srt.layers.attention.hip_flash_mla import (
        _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS,
        _uniform_indptr,
    )

    n_tokens, n_heads, head_dim = q.shape
    if output is None:
        output = torch.empty_like(q)
    device = str(q.device)
    extra = {}
    if extra_kv_cache is not None:
        extra = dict(
            extra_cache=extra_kv_cache.view(torch.uint8).squeeze(2),
            extra_indices=extra_sparse_indices.reshape(-1),
            extra_indptr=_uniform_indptr(n_tokens, extra_sparse_indices.shape[-1], device),
        )
    if n_tokens >= _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS:
        extra["kv_splits"] = 1
    attended = pa_decode_sparse(
        q,
        kv_cache.view(torch.uint8).squeeze(2),
        sparse_indices.reshape(-1),
        _uniform_indptr(n_tokens, sparse_indices.shape[-1], device),
        sinks,
        sm_scale,
        skip_reduce=True,
        out=output,
        **extra,
    )
    rope_dim = rope_freqs.shape[1]
    if isinstance(attended, tuple):
        # split-K partials: sglang's gfx_sparse_split_reduce, into the caller's output
        part_acc, part_m, part_l = attended
        _gfx_sparse_decode_reduce_kernel[(n_tokens, n_heads)](
            part_m,
            part_l,
            part_acc,
            sinks,
            output,
            rope_freqs,
            positions,
            part_m.stride(0),
            part_m.stride(1),
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
            output.stride(0),
            output.stride(1),
            rope_freqs.stride(0),
            NUM_SPLITS=part_acc.shape[1],
            D=head_dim,
            RD=rope_dim,
            HAS_SINK=True,
            INV_ROPE=True,
            num_warps=1,
        )
        return output
    # _apply_inverse_rope: apply_rotary_emb_triton's flat-row kernel, as DeepseekV4 deploys it
    rope_part = output[..., -rope_dim:]
    n_rows = n_tokens * n_heads
    rd = max(triton.next_power_of_2(rope_dim), 2)
    apply_rotary_emb_flat_kernel[(triton.cdiv(n_rows, 16),)](
        rope_part,
        rope_freqs,
        positions,
        n_rows,
        n_heads,
        rope_part.stride(0),
        rope_part.stride(1),
        rope_part.stride(2),
        rope_freqs.stride(0),
        rope_freqs.stride(1),
        USE_POS=True,
        IS_INVERSE=True,
        RD=rd,
        RDH=rd // 2,
        BLOCK_ROWS=16,
        num_warps=1,
    )
    return output
