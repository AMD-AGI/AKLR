# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.forward_mla_rocm.contract import (
    MLA_KC_ROPE_CACHE,
    MLA_QKV_A_NORM,
    MLA_VC,
)

__all__ = ["mla_kc_rope_cache", "mla_qkv_a_norm", "mla_vc"]


@api(MLA_QKV_A_NORM)
def mla_qkv_a_norm(hidden_states, w_qkv_a, q_norm_weight, kv_norm_weight, eps=1e-5): ...


@api(MLA_KC_ROPE_CACHE)
def mla_kc_rope_cache(
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
): ...


@api(MLA_VC)
def mla_vc(attn_output, w_vc, w_scale): ...


@sglang_route(
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm."
    "DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare",
    versions=("0.5.22",),
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
    the op declines it, and when the core's route would serve the fused absorb with
    mla_kc_rope_cache: that op reads the RoPE key inside the whole qkv latent, which this op
    does not produce."""
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
    if fuse_bmm_rope_cache and _core_serves_kc(
        self, hidden_states, positions, forward_batch, llama_4_scaling, rocm
    ):
        return FallbackReason.OUT_OF_CONTRACT
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


def _core_serves_kc(attn, hidden_states, positions, forward_batch, llama_4_scaling, rocm) -> bool:
    """Whether the core's route would hand mla_kc_rope_cache the call sglang's own prepare
    leaves it, asked with empty stand-ins of that prepare's query, qkv latent and KV latent."""
    if _core_falls_back(attn, forward_batch, llama_4_scaling, rocm):
        return False
    tokens, like = hidden_states.shape[0], {"dtype": hidden_states.dtype, "device": hidden_states.device}
    rope_start = attn.q_lora_rank + attn.kv_lora_rank
    q = torch.empty((tokens, attn.num_local_heads, attn.qk_head_dim), **like)
    latent = torch.empty((tokens, rope_start + attn.qk_rope_head_dim), **like)
    k_nope = torch.empty((tokens, 1, attn.kv_lora_rank), **like)
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    kc_arguments = _kc_arguments(
        attn, q_nope, q_pe, k_nope, latent[:, rope_start:].unsqueeze(1), positions, forward_batch
    )
    return kc_arguments is not None and mla_kc_rope_cache.fallback_reason(**kc_arguments) is None


@sglang_route(
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm."
    "DeepseekMLARocmForwardMixin.forward_absorb_rocm_core",
    versions=("0.5.22",),
)
def _forward_absorb_rocm_core(
    self,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    forward_batch,
    zero_allocator,
    positions,
    topk_indices,
    llama_4_scaling,
    q_nope_unabsorbed=None,
):
    """The decode branch of forward_absorb_rocm_core whose query absorb prepare left to it
    (q_nope_unabsorbed, a DSA decode on gfx950): the fused absorb, RoPE and KV write as
    mla_kc_rope_cache, sglang's attention, the value absorb as mla_vc, then o_proj. An op
    that declines the call leaves its step to sglang's own function; with neither taking it,
    sglang runs the whole method."""
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as rocm

    if q_nope_unabsorbed is None or _core_falls_back(self, forward_batch, llama_4_scaling, rocm):
        return FallbackReason.OUT_OF_CONTRACT
    kc_arguments = _kc_arguments(self, q_nope_unabsorbed, q_pe, k_nope, k_pe, positions, forward_batch)
    kc_reason = (
        FallbackReason.OUT_OF_CONTRACT
        if kc_arguments is None
        else mla_kc_rope_cache.fallback_reason(**kc_arguments)
    )
    vc_arguments = _vc_arguments(self, rocm)
    vc_reason = (
        FallbackReason.OUT_OF_CONTRACT
        if vc_arguments is None
        else mla_vc.fallback_reason(attn_output=_attention_stand_in(self, q_pe), **vc_arguments)
    )
    if kc_reason is not None and vc_reason is not None:
        return kc_reason
    # the original's steps from here: nothing returns a FallbackReason after the KV write
    q_cat = None if kc_reason is not None else mla_kc_rope_cache(**kc_arguments)
    if q_cat is None or isinstance(q_cat, FallbackReason):
        q_cat, _, k_rope, _ = rocm._fused_bmm_rope_cat_and_cache(
            self, q_nope_unabsorbed, q_pe, k_nope, k_pe, positions, forward_batch.out_cache_loc
        )
    else:
        # the rotated key is in the KV pool: the attention of a call that saves no KV
        # reads no k_rope
        k_rope = None
    attn_output = self.attn_mqa(
        q_cat,
        None,
        None,
        forward_batch,
        q_rope=None,
        k_rope=k_rope,
        save_kv_cache=False,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )
    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
    attn_bmm_output = None if vc_reason is not None else mla_vc(attn_output=attn_output, **vc_arguments)
    if attn_bmm_output is None or isinstance(attn_bmm_output, FallbackReason):
        attn_bmm_output = rocm.rocm_absorb_v_bmm(self, attn_output)
    output, _ = self.o_proj(attn_bmm_output)
    if self.next_skip_topk is None:
        return output
    # topk_indices for the next layer when the index cache is on
    return (output, topk_indices) if self.next_skip_topk else (output, None)


def _core_falls_back(attn, forward_batch, llama_4_scaling, rocm) -> bool:
    """Whether forward_absorb_rocm_core computes other than the route: another branch than
    the fused RoPE-and-cache one with the absorbed query passed whole to the attention, a
    llama-4 scaling or LoRA on kv_b_proj, DCP, a KV scale (the ops write the pool unscaled)
    or NeoX RoPE pairs (the ops rotate GPT-J's)."""
    mode = forward_batch.forward_mode
    k_scale = attn.attn_mqa.k_scale
    k_scale_float = getattr(attn.attn_mqa, "k_scale_float", None)
    return (
        attn.current_attention_backend not in rocm.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        or attn.rotary_emb is None
        or not attn._skip_rope_for_dsa_tilelang_fused()
        or not (mode.is_decode_or_idle() or mode.is_target_verify())
        or llama_4_scaling is not None
        or attn.dcp_size != 1
        or attn.use_deep_gemm_bmm
        or rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        or rocm.is_kv_b_lora_active(attn)
        or (k_scale is not None and k_scale_float != 1.0)
        or attn.rotary_emb.is_neox_style
    )


def _kc_arguments(attn, q_nope, q_pe, k_nope, k_pe, positions, forward_batch):
    """mla_kc_rope_cache's arguments as sglang holds them, zero-copy; None for a tensor not
    laid out as the projections and the pools leave them."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    rotary = attn.rotary_emb
    kv_cache = get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id)
    arguments = {
        "q": _whole_rows(q_nope, q_pe),
        "qkv_latent": _qkv_latent(k_pe, attn.q_lora_rank, attn.kv_lora_rank),
        "kv_latent": _rows(k_nope),
        "w_kc": attn.w_kc.transpose(1, 2),
        "w_scale": _scale(attn.w_scale),
        "cos_cache": _rows(getattr(rotary, "cos_cache", None)),
        "sin_cache": _rows(getattr(rotary, "sin_cache", None)),
        "positions": positions,
        "out_cache_loc": forward_batch.out_cache_loc,
        "kv_cache": _rows(kv_cache),
    }
    return None if any(value is None for value in arguments.values()) else arguments


def _vc_arguments(attn, rocm):
    """mla_vc's weights as sglang holds them, for rocm_absorb_v_bmm's fp8 branch into a bf16
    o_proj; None where it takes another branch."""
    w_scale = _scale(attn.w_scale)
    if not (
        rocm._use_aiter_gfx95
        and attn.w_kc.dtype == torch.float8_e4m3fn
        and attn.o_proj.weight.dtype != torch.uint8
        and not rocm._is_block_scale_fp8(attn.o_proj)
        and w_scale is not None
    ):
        return None
    return {"w_vc": attn.w_vc.transpose(1, 2), "w_scale": w_scale}


def _rows(tensor):
    """A contiguous tensor as [rows, everything else], a view; None for another tensor."""
    if not (isinstance(tensor, torch.Tensor) and tensor.dim() >= 2 and tensor.is_contiguous()):
        return None
    return tensor.view(tensor.shape[0], -1)


def _scale(w_scale):
    """kv_b_proj's per-tensor scale, sglang's 0-d tensor, as the contracts' one element (a
    view of it); None for another value."""
    return w_scale.view(1) if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 0 else None


def _attention_stand_in(attn, q_pe):
    """An empty tensor shaped as the attention's output, to ask mla_vc before the KV write."""
    return torch.empty(
        (q_pe.shape[0], attn.num_local_heads, attn.kv_lora_rank), dtype=q_pe.dtype, device=q_pe.device
    )


def _whole_rows(first, second):
    """The [tokens, heads, a + b] tensor whose two adjacent views prepare split q into; None
    for two tensors apart."""
    tokens, heads, width = first.shape
    whole = width + second.shape[-1]
    if not (
        second.dtype == first.dtype
        and second.shape[:2] == first.shape[:2]
        and first.stride() == second.stride() == (heads * whole, whole, 1)
        and second.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
        and second.storage_offset() == first.storage_offset() + width
    ):
        return None
    return first.as_strided((tokens, heads, whole), first.stride())


def _qkv_latent(k_pe, q_lora_rank, kv_lora_rank):
    """The [tokens, q_lora_rank + kv_lora_rank + rope] qkv latent whose last columns k_pe
    [tokens, 1, rope] views (fetch_qkv_latent's, split by prepare); None otherwise."""
    tokens, _, rope = k_pe.shape
    offset = q_lora_rank + kv_lora_rank
    width = offset + rope
    if not (k_pe.stride(0) == width and k_pe.stride(-1) == 1 and k_pe.storage_offset() >= offset):
        return None
    return k_pe.as_strided((tokens, width), (width, 1), k_pe.storage_offset() - offset)
