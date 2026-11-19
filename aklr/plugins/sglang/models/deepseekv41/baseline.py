# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23.dev20261111 runs on gfx950 for DeepSeek-V4.1 where no monokernel
serves the step."""

from __future__ import annotations

__all__ = [
    "deepseekv41_mono_k1_baseline",
    "deepseekv41_mono_k2_attention_baseline",
    "deepseekv41_mxfp8_linear_baseline",
    "deepseekv41_sparse_mla_baseline",
]


def deepseekv41_mxfp8_linear_baseline(x, weight, weight_scale, x_scale=None, output=None):
    """aiter_mxfp8_blockscaled_linear as Fp8LinearMethod calls it: sglang's quantization of a
    bf16 x, then aiter's blockscale GEMM at the kernel its tuned tables pick."""
    import torch
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear

    return aiter_mxfp8_blockscaled_linear(
        input=x,
        weight=weight,
        weight_scale_ue8m0=weight_scale.view(torch.uint8),
        input_scale=None if x_scale is None else x_scale.view(torch.uint8),
    )


def deepseekv41_sparse_mla_baseline(
    q,
    kv_cache,
    indices,
    sinks,
    sm_scale,
    extra_kv_cache=None,
    extra_indices=None,
    rope_freqs=None,
    positions=None,
    kv_cache_layout="row_major",
    output=None,
):
    """aiter_sparse_decode_fwd as DeepseekV4HipRadixBackend calls it: aiter's sparse decode
    kernel, split over the KV below 1024 tokens with sglang's reduce folding the rotation in,
    unsplit from there with sglang's rotation after it."""
    from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd

    attended, _ = aiter_sparse_decode_fwd(
        q,
        kv_cache,
        indices,
        sinks,
        sm_scale,
        extra_k_cache=extra_kv_cache,
        extra_indices_in_kvcache=extra_indices,
        inv_rope=None if rope_freqs is None else (rope_freqs, positions),
    )
    return attended


def _boundary(
    x, residual, post_mix, comb_mix, pre_mix, hc_fn, hc_scale, hc_base, sinkhorn_iters, hc_rms_eps, hc_eps
):
    """sglang's gfx950 mHC boundary: (the streams, the layer input, the pending coefficients)."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import hc_boundary_fused_deferred

    streams, layer_input, coefficients = hc_boundary_fused_deferred(
        x,
        residual,
        post_mix,
        comb_mix,
        pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        4,
        sinkhorn_iters,
        hc_rms_eps,
        hc_eps,
    )
    if streams is None:
        streams = residual
    if layer_input is None:
        layer_input = residual[:, 0, :].contiguous()
    return streams, layer_input, coefficients


def deepseekv41_mono_k1_baseline(
    residual,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    wqkv,
    wqkv_scale,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    rope_freqs,
    positions,
    x=None,
    post_mix=None,
    comb_mix=None,
    prev_pre_mix=None,
    hc_rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    sinkhorn_iters=20,
    q=None,
    residual_out=None,
    pre_mix_out=None,
    post_mix_out=None,
    comb_mix_out=None,
    normed=None,
    q_latent=None,
    k_nope=None,
    k_nope_scale=None,
    k_rope=None,
):
    """The decode chain the monokernel's K1 stands in for: the fused boundary, the input norm
    hosting its Sinkhorn and emitting MXFP8, wqkv_a on aiter's blockscale GEMM, q_norm with
    its bf16 output, wq_b quantizing it, sglang's RoPE on q, and the fused KV norm, RoPE and
    record store, read back from a pool of this call's own."""
    import torch
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla, fused_rope_inplace
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn
    from sglang.kernels.ops.layernorm.rmsnorm_fp8_e8m0 import rmsnorm_fp8_e8m0_quant

    streams, layer_input, coefficients = _boundary(
        x,
        residual,
        post_mix,
        comb_mix,
        prev_pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        sinkhorn_iters,
        hc_rms_eps,
        hc_eps,
    )
    quantized, normed = rmsnorm_with_sinkhorn(layer_input, norm_weight, norm_eps, coefficients, emit_fp8=True)
    pre, post, comb = coefficients.tensors()
    qkv = aiter_mxfp8_blockscaled_linear(
        input=quantized.q,
        weight=wqkv,
        weight_scale_ue8m0=wqkv_scale.view(torch.uint8),
        input_scale=quantized.scale,
    )
    q_rank, tokens = q_norm_weight.shape[0], residual.shape[0]
    _, q_latent = rmsnorm_fp8_e8m0_quant(qkv[:, :q_rank].contiguous(), q_norm_weight, norm_eps)
    q = aiter_mxfp8_blockscaled_linear(
        input=q_latent, weight=wq_b, weight_scale_ue8m0=wq_b_scale.view(torch.uint8)
    ).view(tokens, -1, 512)
    freqs_cis = torch.view_as_complex(rope_freqs.view(rope_freqs.shape[0], -1, 2).contiguous())
    fused_rope_inplace(q[..., -64:], None, freqs_cis, positions=positions)
    # one page of 64 records: 576 bytes each, then 8 of scales each, the row padded to 576s
    page = 64
    pages = -(-tokens // page)
    pool = torch.zeros((pages, -(-page * 584 // 576) * 576), dtype=torch.uint8, device=residual.device)
    slots = torch.arange(tokens, dtype=torch.int32, device=residual.device)
    fused_k_norm_rope_flashmla(
        qkv[:, q_rank:].contiguous(), kv_norm_weight, norm_eps, freqs_cis, positions, slots, pool, page
    )
    records = pool[:, : page * 576].reshape(-1, 576)[:tokens]
    scales = pool[:, page * 576 : page * 584].reshape(-1, 8)[:tokens, :7]
    k_nope = records[:, :448].contiguous().view(torch.float8_e4m3fn)
    k_rope = records[:, 448:].contiguous().view(torch.bfloat16)
    return q, streams, pre, post, comb, normed, q_latent, k_nope, scales.contiguous(), k_rope


def deepseekv41_mono_k2_attention_baseline(
    q,
    kv_cache,
    indices,
    sinks,
    sm_scale,
    wo_a,
    wo_b,
    wo_b_scale,
    residual,
    post_mix,
    comb_mix,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    extra_kv_cache=None,
    extra_indices=None,
    rope_freqs=None,
    positions=None,
    kv_cache_layout="row_major",
    hc_rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    sinkhorn_iters=20,
    ffn_input=None,
    residual_out=None,
    pre_mix_out=None,
    post_mix_out=None,
    comb_mix_out=None,
):
    """The decode chain the monokernel's K2 stands in for, up to the MoE: the sparse attention
    with the rotation back, wo_a's bf16 grouped product as a verify step runs it, wo_b on
    aiter's blockscale GEMM, the fused boundary and the post-attention norm hosting its
    Sinkhorn."""
    import torch
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    attended = deepseekv41_sparse_mla_baseline(
        q, kv_cache, indices, sinks, sm_scale, extra_kv_cache, extra_indices, rope_freqs, positions
    )
    tokens, groups = q.shape[0], q.shape[2] // 8
    grouped, weight = attended.reshape(tokens, groups, -1), wo_a.view(groups, -1, wo_a.shape[-1])
    # _apply_wo_a_bf16_matmul's verify rows on gfx950: its fork of aiter's batched GEMM, else einsum
    if gfx95_dense._wo_a_fp8_grid_gemm is None:
        projected = torch.einsum("tgd,grd->tgr", grouped, weight)
    else:
        projected = gfx95_dense._wo_a_fp8_grid_gemm(grouped, weight, fp8_grid=False, split_k=None)
    x = aiter_mxfp8_blockscaled_linear(
        input=projected.reshape(tokens, -1), weight=wo_b, weight_scale_ue8m0=wo_b_scale.view(torch.uint8)
    )
    streams, layer_input, coefficients = _boundary(
        x, residual, post_mix, comb_mix, pre_mix, hc_fn, hc_scale, hc_base, sinkhorn_iters, hc_rms_eps, hc_eps
    )
    _, ffn_input = rmsnorm_with_sinkhorn(layer_input, norm_weight, norm_eps, coefficients, fake_quant=False)
    pre, post, comb = coefficients.tensors()
    return ffn_input, streams, pre, post, comb
