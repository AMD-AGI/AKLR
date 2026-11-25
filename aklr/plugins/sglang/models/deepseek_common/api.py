# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_common.contract import MLA_QKV_A_NORM

__all__ = ["mla_qkv_a_norm"]


@api(MLA_QKV_A_NORM)
def mla_qkv_a_norm(hidden_states, w_qkv_a, q_norm_weight, kv_norm_weight, eps=1e-5): ...


@sglang_route(
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm."
    "DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare",
    versions=("0.5.22", "0.5.23"),
)
def _forward_absorb_rocm_prepare(
    self,
    positions,
    hidden_states,
    forward_batch,
    zero_allocator,
    llama_4_scaling=None,
    prev_topk_indices=None,
):
    """forward_absorb_rocm_prepare where its compressed-query branch normalizes the qkv latent
    with aiter's fused bf16 RMSNorm (a bf16 q_b_proj): mla_qkv_a_norm for the qkv latent's
    projection, split and norms, then the original's steps (q_b_proj, the indexer, the query
    absorb unless the core fuses it, RoPE where prepare applies it). sglang keeps the call when
    the op declines it."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as rocm

    if _prepare_falls_back(self, hidden_states, rocm):
        return FallbackReason.OUT_OF_CONTRACT
    arguments = {
        "hidden_states": hidden_states,
        "w_qkv_a": self.fused_qkv_a_proj_with_mqa.weight,
        "q_norm_weight": self.q_a_layernorm.weight,
        "kv_norm_weight": self.kv_a_layernorm.weight,
        "eps": self.q_a_layernorm.variance_epsilon,
    }
    reason = mla_qkv_a_norm.fallback_reason(**arguments)
    if reason is not None:
        return reason
    fuse_bmm_rope_cache = (
        not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        and rocm._can_fuse_bmm_rope_cat_and_cache(self)
    )
    result = mla_qkv_a_norm(**arguments)
    if isinstance(result, FallbackReason):
        return result
    # the original's steps from here, on its branch without an alt stream, DCP, the deep-gemm
    # BMM or a LoRA: nothing returns a FallbackReason after the op
    q, k_nope, k_rope = result
    q_lora = q if self.use_dsa else None
    topk_indices = None
    k_nope = k_nope.unsqueeze(1)
    q = self.q_b_proj_forward(q)
    if q_lora is not None:
        if self.should_run_indexer(prev_topk_indices):
            topk_indices = self.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )
        else:
            topk_indices = rocm.maybe_capture_indexer_topk(self.layer_id, prev_topk_indices)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # the RoPE key in a tensor of its own, not a view of the qkv latent
    k_pe = k_rope.unsqueeze(1)
    if fuse_bmm_rope_cache:
        q_nope_out = None
    else:
        q_nope_out = rocm.rocm_absorb_q_bmm(self, q_nope, is_capture_mode=get_is_capture_mode())
        q_nope_out = q_nope_out.transpose(0, 1)
    fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
    # the original's DCP decode term is false without DCP
    if self.rotary_emb is not None and (
        (not fuse_rope_for_trtllm_mla)
        and (not self._skip_rope_for_dsa_tilelang_fused())
        and (not self._skip_rope_for_aiter_fused_mla())
        and (
            not rocm._use_aiter
            or not rocm._is_gfx95_supported
            or self.use_dsa
            or (
                self.current_attention_backend not in rocm.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
                and self.current_attention_backend != "aiter"
            )
        )
    ):
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        q_nope if fuse_bmm_rope_cache else None,
    )


def _prepare_falls_back(attn, hidden_states, rocm) -> bool:
    """Whether forward_absorb_rocm_prepare computes other than the route: no compressed query
    or no fused qkv_a projection (prepare_qkv_latent's other branch), a quantized, biased or
    LoRA projection, a qkv latent sglang already holds, would gather (input-scattered
    attention) or projects from other hidden states, another norm branch than aiter's fused
    bf16 one (the alt stream, an mxfp4 or block-fp8 q_b_proj, aiter off), two epsilons, DCP, the
    deep-gemm BMM or a kv_b LoRA."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.runtime_context import get_forward, get_lora

    projection = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    inputs = get_forward().attn_inputs
    return (
        attn.q_lora_rank is None
        or projection is None
        or hasattr(attn, "q_a_proj")
        or type(projection.quant_method) is not UnquantizedLinearMethod
        or projection.bias is not None
        or get_lora().enable_lora
        or rocm.get_attn_tp_context().input_scattered
        or inputs is None
        or inputs.hidden_states_local is not hidden_states
        or inputs.qkv_latent_ is not None
        or attn.alt_stream is not None
        or not rocm._use_aiter
        or (rocm._use_aiter_gfx95 and attn.q_b_proj.weight.dtype == torch.uint8)
        or (rocm._use_aiter_gfx95 and rocm._is_block_scale_fp8(attn.q_b_proj))
        or attn.q_a_layernorm.variance_epsilon != attn.kv_a_layernorm.variance_epsilon
        or attn.dcp_size != 1
        or attn.use_deep_gemm_bmm
        or rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        or rocm.is_kv_b_lora_active(attn)
    )
