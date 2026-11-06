# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = [
    "deepseek_v4_attention_q_prep_baseline",
    "deepseek_v4_sparse_mla_decode_inv_rope_baseline",
    "deepseek_v4_wo_a_split_k_baseline",
    "deepseek_v4_wo_a_split_k_mxfp8_baseline",
]


def deepseek_v4_attention_q_prep_baseline(
    qkv_a,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    freqs,
    positions,
    swa_loc,
    kv_cache,
    q,
    page_size=256,
    q_eps=9.99999968265522e-21,
    kv_eps=9.99999968265522e-21,
    q_lora_quant=None,
    q_lora_scale=None,
):
    """What sglang 0.5.23's MQALayer._forward_prepare runs on gfx950 for this arm, here writing
    the caller's outputs instead of new tensors: q_norm's rmsnorm_fake_quant_fp8 kernel (with
    the BF16 norm it also writes), wq_b on aiter's group-32 Triton GEMM (sglang's aiter MXFP8
    route below its FlyDSL threshold of 2048 rows), then the K store
    fused_k_norm_rope_flashmla roping q in place."""
    import torch
    import triton
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import gemm_a8w8_blockscale_group32
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla
    from sglang.kernels.ops.layernorm.rmsnorm_fake_quant_amd_gfx95 import (
        _rmsnorm_fake_quant_fp8_kernel,
        rmsnorm_row_chunk,
    )
    from sglang.srt.utils import is_gfx95_supported

    if not is_gfx95_supported():
        raise NotImplementedError("sglang runs this arm on gfx950 only")
    n_tokens, rank = qkv_a.shape[0], q_norm_weight.shape[0]
    device = qkv_a.device
    if q_lora_quant is None:
        q_lora_quant = torch.empty((n_tokens, rank), dtype=torch.float8_e4m3fn, device=device)
    if q_lora_scale is None:
        q_lora_scale = torch.empty((n_tokens, rank // 32), dtype=torch.float8_e8m0fnu, device=device)
    q_lora = qkv_a[:, :rank]
    norm = torch.empty((n_tokens, rank), dtype=qkv_a.dtype, device=device)
    scale = q_lora_scale.view(torch.uint8)
    chunk = rmsnorm_row_chunk(rank)
    _rmsnorm_fake_quant_fp8_kernel[(n_tokens,)](
        q_lora,
        q_norm_weight,
        q_lora,
        q_lora_quant,
        norm,
        scale,
        rank,
        q_lora.stride(0),
        0,
        q_lora_quant.stride(0),
        norm.stride(0),
        scale.stride(0),
        q_eps,
        HAS_RESIDUAL=False,
        WRITE_NORM=True,
        EMIT_FP8=True,
        CHUNK=chunk,
        NUM_CHUNKS=triton.cdiv(rank, chunk),
        num_warps=4,
    )
    gemm_a8w8_blockscale_group32(
        q_lora_quant,
        wq_b,
        scale,
        wq_b_scale.view(torch.uint8),
        dtype=torch.bfloat16,
        y=q.view(n_tokens, -1),
        weight_group_rows=32,
    )
    fused_k_norm_rope_flashmla(
        kv=qkv_a[:, rank:],
        kv_weight=kv_norm_weight,
        eps=kv_eps,
        freqs_cis=torch.view_as_complex(freqs.view(freqs.shape[0], -1, 2)),
        positions=positions,
        out_loc=swa_loc,
        kvcache=kv_cache,
        page_size=page_size,
        q=q,
    )
    return q, q_lora_quant, q_lora_scale


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


def _wo_a_split_k(x, w, out, fp8_grid, scale):
    import inspect

    from sglang.kernels.ops.gemm import gfx95_batched_gemm_bf16_fp8_grid as gemm

    n_tokens, _, in_dim = x.shape
    if not gemm._split_k_applies(n_tokens, in_dim, w.shape[1]):
        raise NotImplementedError("outside sglang's split-K regime")
    # sglang's own function, also where the route hooks it
    inspect.unwrap(gemm._batched_gemm_split_k)(x, w, out, fp8_grid, scale)


def deepseek_v4_wo_a_split_k_baseline(x, w, fp8_grid=False, out_dtype=None, out=None):
    """sglang 0.5.23's gfx950 split-K wo_a GEMM, _batched_gemm_split_k: the K-split partial
    kernel into a float32 scratch, then the reduce kernel (split order, bf16, fp8 grid), into
    the caller's out."""
    import torch

    if out is None:
        out = torch.empty((x.shape[0], w.shape[0] * w.shape[1]), dtype=torch.bfloat16, device=x.device)
    _wo_a_split_k(x, w, out, fp8_grid, None)
    return out


def deepseek_v4_wo_a_split_k_mxfp8_baseline(x, w, quantized=None, scales=None):
    """sglang 0.5.23's gfx950 split-K wo_a GEMM, _batched_gemm_split_k, emitting fp8: the
    K-split partial kernel, then the reduce kernel quantizing the bf16 sum, into the caller's
    quantized and scales (the e8m0 bytes, as sglang's uint8)."""
    import torch

    shape = (x.shape[0], w.shape[0] * w.shape[1])
    if quantized is None:
        quantized = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    if scales is None:
        scales = torch.empty((shape[0], shape[1] // 32), dtype=torch.float8_e8m0fnu, device=x.device)
    _wo_a_split_k(x, w, quantized, False, scales.view(torch.uint8))
    return quantized, scales
