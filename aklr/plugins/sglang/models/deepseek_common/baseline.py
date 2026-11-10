# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on aiter on gfx950 for these steps of
DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare and forward_absorb_rocm_core, kv_b_proj
in fp8 at one scale per tensor (GLM-5.2's, which sglang quantizes at load): the aiter kernels
forward_mla_rocm binds, called as it calls them."""

from __future__ import annotations

from types import SimpleNamespace

from aklr.plugins.sglang.models.deepseek_common.contract import FP8_GROUP

__all__ = ["mla_kc_rope_cache_baseline", "mla_qkv_a_norm_baseline", "mla_vc_baseline"]


def _forward_mla_rocm():
    """sglang's forward_mla_rocm; raise unless it took the aiter gfx950 path the baselines
    replay, chosen at import."""
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm

    if not forward_mla_rocm._use_aiter_gfx95:
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")
    return forward_mla_rocm


def mla_qkv_a_norm_baseline(
    hidden_states,
    w_qkv_a,
    q_norm_weight,
    kv_norm_weight,
    eps=1e-5,
    q_latent=None,
    kv_latent=None,
    k_rope=None,
):
    """The qkv latent prepare fetches and its norms, for a bf16 q_b_proj: fused_qkv_a_proj_with_mqa
    through sglang's UnquantizedLinearMethod (aiter's tuned GEMM at these shapes), split as
    prepare splits it, then the fused q and KV RMSNorm it binds (fused_qk_rmsnorm_bf16). The RoPE
    key is the qkv latent's last columns, the view prepare hands on."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    rocm = _forward_mla_rocm()
    q_rank, kv_rank = q_norm_weight.shape[0], kv_norm_weight.shape[0]
    latent = UnquantizedLinearMethod().apply(SimpleNamespace(weight=w_qkv_a), hidden_states)
    q, latent_cache = latent.split([q_rank, latent.shape[-1] - q_rank], dim=-1)
    q, kv = rocm.fused_qk_rmsnorm_bf16(
        q, q_norm_weight, eps, latent_cache[..., :kv_rank], kv_norm_weight, eps
    )
    return q, kv, latent_cache[..., kv_rank:]


def mla_kc_rope_cache_baseline(
    q,
    qkv_latent,
    kv_latent,
    w_kc,
    w_scale,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    kv_cache,
    q_cat=None,
):
    """_fused_bmm_rope_cat_and_cache as forward_absorb_rocm_core calls it in a DSA decode:
    aiter's fused fp8 BMM, RoPE and KV write, q in fp8 for the fp8 pool, no k_scale (the
    layer has none). sglang hands it w_kc.transpose(-1, -2), these bytes, and its 0-d
    w_scale, this one element viewed back."""
    rocm = _forward_mla_rocm()
    nope, rope = w_kc.shape[-1], cos_cache.shape[-1] * 2
    out, _, _, _ = rocm.fused_fp8_bmm_rope_cat_and_cache_mla(
        q[..., :nope].transpose(0, 1),
        w_kc,
        w_scale.view(()),
        q[..., nope:],
        kv_latent.unsqueeze(1),
        qkv_latent[:, -rope:].unsqueeze(1),
        kv_cache.unsqueeze(1),
        out_cache_loc,
        positions,
        cos_cache,
        sin_cache,
        group_size=FP8_GROUP,
        k_scale=None,
        is_neox=False,
        q_out_dtype=kv_cache.dtype,
    )
    return out


def mla_vc_baseline(attn_output, w_vc, w_scale, attn_bmm_output=None):
    """rocm_absorb_v_bmm for fp8 w_vc and a bf16 o_proj: aiter's a8w8 batched GEMM with
    per-token-group prequant into [tokens, heads, v_head_dim], at sglang's decode tile
    config, then flattened. sglang hands it w_vc.transpose(-1, -2), these bytes, and its 0-d
    w_scale, this one element viewed back."""
    import torch

    rocm = _forward_mla_rocm()
    tokens, heads, _ = attn_output.shape
    v_head_dim = w_vc.shape[1]
    out = torch.empty(tokens, heads, v_head_dim, device=attn_output.device, dtype=torch.bfloat16)
    rocm.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        X=attn_output,
        WQ=w_vc,
        w_scale=w_scale.view(()),
        group_size=FP8_GROUP,
        YQ=out,
        transpose_bm=True,
        transpose_bm_in=True,
        dtype=torch.bfloat16,
        config=rocm._absorb_bmm_config(heads, tokens, v_head_dim),
    )
    return out.flatten(1, 2)
