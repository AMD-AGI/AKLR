# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 and 0.5.23 run on aiter on gfx950 for these steps of
DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare and forward_absorb_rocm_core, kv_b_proj
in fp8 at one scale per tensor (GLM-5.2's, which sglang quantizes at load): the aiter kernels
forward_mla_rocm binds, called as it calls them."""

from types import SimpleNamespace

from aklr.core.contract.baseline import OpBaseline

__all__ = ["MlaQkvANormBaseline"]


def _forward_mla_rocm():
    """sglang's forward_mla_rocm; raise unless it took the aiter gfx950 path the baselines
    replay, chosen at import."""
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm

    if not forward_mla_rocm._use_aiter_gfx95:
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")
    return forward_mla_rocm


class MlaQkvANormBaseline(OpBaseline):
    def baseline(
        self,
        *,
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
        return {"q_latent": q, "kv_latent": kv, "k_rope": latent_cache[..., kv_rank:]}
