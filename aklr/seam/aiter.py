###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import os

import torch

from aklr.ops.attention import paged_attention_decode, unified_attention
from aklr.ops.topk import biased_grouped_topk
from aklr.ops.fused_moe import fused_moe
from aklr.ops.fused_moe.baseline import aiter_moe_act_quant
from aklr.ops.gemm import (
    bmm_bf16_nt,
    gemm_a8w8_nt,
    gemm_bf16_nt,
    gemm_fp8_nt_blockscaled,
    gemm_mxfp4_nt,
)
from aklr.ops.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from aklr.ops.mla import fp8_paged_mqa_logits
from aklr.ops.mla_prefill import sparse_prefill_mxfp8
from aklr.ops.quant import per_group_quant_fp8, per_group_quant_mxfp8
from aklr.ops.rms_norm import add_rms_norm, rms_norm
from aklr.seam import adapter


@adapter(
    "aiter.ops.topk.biased_grouped_topk",
    biased_grouped_topk,
)
def _(
    gating_output,
    correction_bias,
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    need_renorm,
    routed_scaling_factor=1.0,
):
    return {
        "router_logits": gating_output,
        "correction_bias": correction_bias,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "topk": topk_ids.shape[-1],
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "renormalize": need_renorm,
        "routed_scaling_factor": routed_scaling_factor,
    }


@adapter("aiter.fused_moe.fused_moe", fused_moe)
def _(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask=None,
    activation=None,
    quant_type=None,
    doweight_stage1=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_size_M=None,
    num_local_tokens=None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode=None,
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
    shared_expert_id=-1,
    stage2_scatter=None,
):
    # moe_sorting_dispatch_policy and splitk are scheduling: the registry decides.
    if block_size_M is not None or hidden_pad or intermediate_pad:
        return None
    if shared_expert_id != -1 or any(
        s is not None for s in (shared_w1, shared_w2, shared_w1_scale, shared_w2_scale)
    ):
        return None
    if stage2_scatter is not None:
        return None  # writes stage 2 elsewhere, on the gfx1250 path
    if any(s is not None for s in (a1_scale, a2_scale, bias1, bias2)):
        return None
    if doweight_stage1 or getattr(quant_type, "name", None) != "per_1x32":
        return None  # the schemas weight in stage 2 and quantize per 1 x 32
    if dtype is not None and dtype != hidden_states.dtype:
        return None
    activation = "silu" if activation is None else activation.name.lower()
    if activation == "situv2" and os.environ.get("AITER_SITUV2_A8W4") != "1":
        return None  # situv2's activation quant is chosen by aiter's environment
    swiglu_limit = swiglu_limit or None  # aiter clamps only at a truthy limit; vLLM passes 0.0 for none
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weight,
        "topk_ids": topk_ids,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "activation": activation,
        "act_quant": aiter_moe_act_quant(activation, swiglu_limit, hidden_states.shape[0]),
        "expert_mask": expert_mask,
        "beta": beta,
        "linear_beta": linear_beta,
        "swiglu_limit": swiglu_limit,
        "num_local_tokens": num_local_tokens,
        "gate_mode": getattr(gate_mode, "value", gate_mode) or "separated",
        "use_shuffled_weight": getattr(w1, "is_shuffled", False) is True,
    }


@adapter("aiter.tuned_gemm.gemm_a16w16", gemm_bf16_nt)
def _(A, B, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
    if any(s is not None for s in (scale_a, scale_b, scale_c)):
        return None
    if A.dtype != torch.bfloat16:
        return None
    return {
        "a": A,
        "b": B,
        "bias": bias,
        "out_dtype": otype or A.dtype,
        "use_shuffled_weight": getattr(B, "is_shuffled", False) is True,
    }


@adapter("aiter.ops.gemm_op_a8w8.gemm_a8w8", gemm_a8w8_nt)
def _(XQ, WQ, x_scale, w_scale, bias=None, dtype=None, splitK=None):
    # splitK is scheduling; the registry decides.
    return {
        "a": XQ,
        "b": WQ,
        "a_scale": x_scale,
        "b_scale": w_scale,
        "bias": bias,
        "out_dtype": dtype,
    }


# CK precedes Triton: the bench baseline is the first adapter that binds, and plain blockscale's is CK.


@adapter(
    "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale",
    gemm_fp8_nt_blockscaled,
)
def _(XQ, WQ, x_scale, w_scale, dtype=None, isBpreshuffled=False):
    if isBpreshuffled:
        return None  # a different gfx950 ASM packing, not the 16x16 one
    return {
        "a": XQ,
        "b": WQ,
        "a_scale": x_scale,
        "b_scale": w_scale,
        "scale_major_mode": "K",
        "out_dtype": dtype,
    }


@adapter(
    "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale.gemm_a8w8_blockscale",
    gemm_fp8_nt_blockscaled,
)
def _(
    x,
    w,
    x_scale,
    w_scale,
    dtype=None,
    y=None,
    config=None,
    skip_reduce=False,
    kernel_type="bandwidth_bound",
    backend=None,
):
    if config is not None or skip_reduce or backend is not None:
        return None  # explicit tuning
    if kernel_type != "bandwidth_bound":
        return None
    return {
        "a": x,
        "b": w,
        "a_scale": x_scale,
        "b_scale": w_scale,
        "scale_major_mode": "K",
        "out_dtype": dtype,
        "out": y,
    }


@adapter(
    "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle",
    gemm_fp8_nt_blockscaled,
)
def _(XQ, WQ, x_scale, w_scale, dtype=None, out=None):
    return {
        "a": XQ,
        "b": WQ,
        "a_scale": x_scale.mT,  # aiter takes the MN-major a_scale
        "b_scale": w_scale,
        "scale_major_mode": "MN",
        "use_shuffled_weight": True,
        "out_dtype": dtype,
        "out": out,
    }


@adapter("aiter.ops.gemm_op_a4w4.gemm_a4w4", gemm_mxfp4_nt)
def _(
    A,
    B,
    A_scale,
    B_scale,
    bias=None,
    dtype=None,
    alpha=1.0,
    beta=0.0,
    bpreshuffle=True,
    apreshuffle=False,
    global_A_scale=None,
    global_B_scale=None,
):
    if not bpreshuffle or apreshuffle:
        return None  # aiter's one layout: b and both scales shuffled
    if global_A_scale is not None or global_B_scale is not None:
        return None  # nvfp4, another op
    if alpha != 1.0 or beta != 0.0:
        return None
    return {
        "a": A,
        "b": B,
        "a_descale": A_scale,
        "b_descale": B_scale,
        "bias": bias,
        "out_dtype": dtype,
        "use_shuffled_weight": True,
    }


@adapter(
    "aiter.ops.batched_gemm_op_bf16.batched_gemm_bf16_CK",
    bmm_bf16_nt,
)
def _(XQ, WQ, bias=None, dtype=None, splitK=None):
    # splitK is scheduling; the registry decides.
    if bias is not None:
        return None
    if XQ.dtype != torch.bfloat16:
        return None
    return {"A": XQ, "B": WQ, "out_dtype": dtype}


@adapter("aiter.ops.mhc.mhc_pre", mhc_pre)
def _(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=1.0,
    sinkhorn_repeat=20,
    norm_weight=None,
    norm_eps=1e-6,
    large_m_splitk=False,
    is_fn_pack_bf16=0,
):
    # large_m_splitk is scheduling, not the schema's.
    if is_fn_pack_bf16:
        return None  # packed projection storage is not yet supported
    return {
        "residual": residual,
        "proj_weight": fn,
        "mix_scale": hc_scale,
        "mix_bias": hc_base,
        "rms_eps": rms_eps,
        "pre_eps": hc_pre_eps,
        "sinkhorn_eps": hc_sinkhorn_eps,
        "post_multiplier": hc_post_mult_value,
        "sinkhorn_iters": sinkhorn_repeat,
        "norm_weight": norm_weight,
        "norm_eps": norm_eps,
    }


@adapter("aiter.ops.mhc.mhc_post", mhc_post)
def _(out, x, residual, post_layer_mix, comb_res_mix, store_nt=-1):
    # store_nt is a store/cache strategy, not semantics.
    return {
        "x": x,
        "residual": residual,
        "post_mix": post_layer_mix.view(*residual.shape[:2], 1),
        "comb_mix": comb_res_mix,
        "out": out,
    }


@adapter(
    "aiter.ops.mhc.mhc_fused_post_pre",
    mhc_fused_post_pre,
)
def _(
    layer_input,
    residual_in,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=1.0,
    sinkhorn_repeat=20,
    norm_weight=None,
    norm_eps=1e-6,
    force_fused=False,
    is_fn_pack_bf16=0,
):
    # force_fused is scheduling, not the schema's.
    if is_fn_pack_bf16:
        return None  # packed projection storage is not yet supported
    return {
        "x": layer_input,  # aiter's input; the schema's layer_input is an output
        "residual": residual_in,
        "post_mix": post_layer_mix.view(*residual_in.shape[:2], 1),
        "comb_mix": comb_res_mix,
        "proj_weight": fn,
        "mix_scale": hc_scale,
        "mix_bias": hc_base,
        "rms_eps": rms_eps,
        "pre_eps": hc_pre_eps,
        "sinkhorn_eps": hc_sinkhorn_eps,
        "post_multiplier": hc_post_mult_value,
        "sinkhorn_iters": sinkhorn_repeat,
        "norm_weight": norm_weight,
        "norm_eps": norm_eps,
    }


@adapter(
    "aiter.ops.rmsnorm.rmsnorm2d_fwd_with_add",
    add_rms_norm,
)
def _(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    gemma_norm=False,
    use_model_sensitive_rmsnorm=0,
):
    if gemma_norm or use_model_sensitive_rmsnorm:
        return None
    return {
        "x": input,
        "residual": residual_in,
        "weight": weight,
        "epsilon": epsilon,
        "normalized": out,
        "residual_out": residual_out,
    }


@adapter("aiter.ops.rmsnorm.rmsnorm2d_fwd", rms_norm)
def _(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
    if use_model_sensitive_rmsnorm:
        return None
    return {"x": input, "weight": weight, "epsilon": epsilon}


@adapter(
    "aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits",
    fp8_paged_mqa_logits,
)
def _(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    Preshuffle=False,
    KVBlockSize=1,
    ChunkK=256,
    TotalCuCount=None,
    WavePerEU=2,
    VarCtxSchedule=None,
):
    # logits' width carries max_model_len; the rest is tiling
    return {
        "q": q_fp8,
        "kv_cache": kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_tables": kv_indices,
        "logits": out_logits,
        "page_size": KVBlockSize,
        "preshuffled": Preshuffle,
    }


# The quant and sparse prefill adapters give the bench its baseline only: their apis return
# other layouts and dtypes than the aiter entry points, which callers would then read.


@adapter("aiter.ops.quant.per_group_quant_hip", per_group_quant_fp8)
def _(*_, **__):
    return None


@adapter("aiter.ops.quant.per_1x32_mx_quant_hip", per_group_quant_mxfp8)
def _(*_, **__):
    return None


@adapter(
    "aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus",
    sparse_prefill_mxfp8,
)
def _(*_, **__):
    return None


@adapter(
    "aiter.ops.triton.attention.unified_attention.unified_attention",
    unified_attention,
)
def _(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache=False,
    skip_reduce=False,
    backend=None,
):
    # max_seqlen_k bounds the kernel's loop; the block table and seqused_k carry the context
    if any(x is not None for x in (q_scales, alibi_slopes, output_scale, qq_bias, sinks)):
        return None
    if shuffled_kv_cache or skip_reduce:
        return None
    _, block_size, heads, dim = k.shape
    split = (block_size * heads * 2 * dim, 2 * dim, block_size * 2 * dim, 1)
    if k.is_contiguous() and v.is_contiguous():
        storage = "row_major"
    elif k.stride() == v.stride() == split and v.data_ptr() == k.data_ptr() + dim * k.element_size():
        storage = "kv_split_pages"  # vLLM's: k and v the halves of each row of one buffer
    else:
        return None
    return {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "softmax_scale": softmax_scale,
        "max_seqlen_q": int(max_seqlen_q),
        "window_size": (int(window_size[0]), int(window_size[1])),
        "causal": bool(causal),
        "kv_cache_storage": storage,
        "softcap": float(softcap),
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "out": out,
    }


@adapter("aiter.ops.attention.paged_attention_v1", paged_attention_decode)
def _(*_, **__):
    return None
