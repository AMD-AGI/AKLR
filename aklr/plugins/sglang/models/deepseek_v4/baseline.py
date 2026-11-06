# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["deepseek_v4_attention_q_prep_baseline"]


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
