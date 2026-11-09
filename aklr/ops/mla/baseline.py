# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["flash_mla_with_kvcache_baseline", "fp8_paged_mqa_logits_baseline"]


def flash_mla_with_kvcache_baseline(
    q,
    kv_cache,
    sparse_indices=None,
    *,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_k=None,
    causal=False,
    head_dim_v=None,
    sm_scale=None,
    sinks=None,
    kv_cache_format="dsv4_fp8",
    kv_scale=1.0,
    return_lse=False,
    kv_cache_layout="row_major",
    rope_freqs=None,
    positions=None,
    output=None,
    lse=None,
):
    """Sparse with the inverse RoPE: sglang's aiter decode (_inv_rope_baseline). Sparse:
    sglang's kernel, as the route calls it. Dense: aiter's Gluon MLA decode, the kernel
    sglang serves the DSPARK verify prefix with, over the 2-D block table read up to
    cache_seqlens or over the flat one cut by cu_seqlens_k; several causal queries per row
    take its multi-token path. It writes into the caller's output and lse when it passes
    them."""
    if rope_freqs is not None:
        if kv_scale != 1.0:
            raise NotImplementedError("sglang's aiter decode takes no kv_scale")
        return _inv_rope_baseline(
            q,
            kv_cache,
            sparse_indices,
            extra_kv_cache,
            extra_sparse_indices,
            sm_scale,
            sinks,
            rope_freqs,
            positions,
            output,
            lse,
        )
    if kv_cache_format == "dsv4_fp8":
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            dpsk_v4_fp8_attention_fwd,
        )

        return dpsk_v4_fp8_attention_fwd(
            q,
            kv_cache,
            None,
            None,
            None,
            None,
            softmax_scale=sm_scale,
            indices=sparse_indices,
            attn_sink=sinks,
            extra_k_cache=extra_kv_cache,
            extra_indices_in_kvcache=extra_sparse_indices,
            topk_length=sparse_lens,
            extra_topk_length=extra_sparse_lens,
        )
    if block_table is None or sinks is not None:
        raise ValueError("dense baseline: a block table and no sinks")
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    batch, query_length, n_heads, dim = q.shape
    if query_length != 1 and not causal:
        raise ValueError("dense baseline: several queries per row only causally")
    if output is None:
        output = q.new_empty(batch, query_length, n_heads, head_dim_v)
    q_nope, q_pe, out = q[..., :head_dim_v], q[..., head_dim_v:], output
    if query_length == 1:
        # one query per row runs the plain decode; several, the causal multi-token path
        q_nope, q_pe = q_nope[:, 0], q_pe[:, 0]
        out = output.view(batch, n_heads, head_dim_v)
    _, row_lse = mla_gluon(
        q_nope,
        q_pe,
        kv_cache.view(-1, dim),
        out,
        block_table,
        cache_seqlens if cu_seqlens_k is None else cu_seqlens_k,
        sm_scale,
        kv_pe_offset=head_dim_v,
        use_2d_view=cu_seqlens_k is None,
        kv_scale=kv_scale,
        return_lse=True,
    )
    row_lse = row_lse.reshape(batch, query_length, n_heads)
    if lse is None:
        lse = row_lse
    else:
        lse.copy_(row_lse)
    return (output, lse) if return_lse else output


def _inv_rope_baseline(
    q,
    kv_cache,
    sparse_indices,
    extra_kv_cache,
    extra_sparse_indices,
    sm_scale,
    sinks,
    rope_freqs,
    positions,
    output,
    lse,
):
    """What sglang 0.5.23's aiter_sparse_decode_fwd runs for DeepSeek-V4's fp8 pools with
    inv_rope outside deterministic inference: aiter's gfx950 pa_decode_sparse (its own split
    count below 1024 rows, one split from 1024), then on one split sglang's flat-row inverse
    RoPE in place over the output's last 64 lanes, or on several sglang's split-K combine with
    the rotation folded in; here writing the caller's output instead of a new tensor. Like
    sglang, it computes no lse (return_lse is false: lse is empty)."""
    import torch
    import triton
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
    from sglang.kernels.ops.attention.dsv4.rope import apply_rotary_emb_flat_kernel
    from sglang.kernels.ops.attention.sparse_decode_reduce_hip import _gfx_sparse_decode_reduce_kernel
    from sglang.srt.layers.attention.hip_flash_mla import (
        _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS,
        _uniform_indptr,
    )

    batch, _, n_heads, head_dim = q.shape
    if output is None:
        output = torch.empty_like(q)
    if lse is None:
        lse = torch.empty((batch, 1, 0), dtype=torch.float32, device=q.device)
    q3, out3 = q.view(batch, n_heads, head_dim), output.view(batch, n_heads, head_dim)
    device = str(q.device)
    extra = {}
    if extra_kv_cache is not None:
        extra = dict(
            extra_cache=extra_kv_cache.view(torch.uint8).squeeze(2),
            extra_indices=extra_sparse_indices.reshape(-1),
            extra_indptr=_uniform_indptr(batch, extra_sparse_indices.shape[-1], device),
        )
    if batch >= _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS:
        extra["kv_splits"] = 1
    attended = pa_decode_sparse(
        q3,
        kv_cache.view(torch.uint8).squeeze(2),
        sparse_indices.reshape(-1),
        _uniform_indptr(batch, sparse_indices.shape[-1], device),
        sinks,
        sm_scale,
        skip_reduce=True,
        out=out3,
        **extra,
    )
    rope_dim = rope_freqs.shape[1]
    if isinstance(attended, tuple):
        # split-K partials: sglang's gfx_sparse_split_reduce, into the caller's output
        part_acc, part_m, part_l = attended
        _gfx_sparse_decode_reduce_kernel[(batch, n_heads)](
            part_m,
            part_l,
            part_acc,
            sinks,
            out3,
            rope_freqs,
            positions,
            part_m.stride(0),
            part_m.stride(1),
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
            out3.stride(0),
            out3.stride(1),
            rope_freqs.stride(0),
            NUM_SPLITS=part_acc.shape[1],
            D=head_dim,
            RD=rope_dim,
            HAS_SINK=True,
            INV_ROPE=True,
            num_warps=1,
        )
        return output, lse
    # _apply_inverse_rope: apply_rotary_emb_triton's flat-row kernel, as DeepseekV4 deploys it
    rope_part = out3[..., -rope_dim:]
    n_rows = batch * n_heads
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
    return output, lse


def fp8_paged_mqa_logits_baseline(
    q, kv_cache, weights, context_lens, block_tables, logits, *, page_size, preshuffled
):
    """aiter's kernel, with the max_model_len the route drops: the logits' width."""
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        logits,
        context_lens,
        block_tables,
        logits.shape[1],
        Preshuffle=preshuffled,
        KVBlockSize=page_size,
    )
    return logits
