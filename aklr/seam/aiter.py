###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import os

import torch

from aklr.ops.biased_grouped_topk import biased_grouped_topk
from aklr.ops.fused_moe import fused_moe
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


def _biased_grouped_topk_baseline(
    original,
    router_logits,
    correction_bias,
    renormalize,
    routed_scaling_factor=1.0,
    topk_weights=None,
    topk_ids=None,
    num_expert_group=1,
    topk_group=1,
    **_,
):
    return lambda: original(
        gating_output=router_logits,
        correction_bias=correction_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        need_renorm=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


@adapter(
    "aiter.ops.topk.biased_grouped_topk",
    biased_grouped_topk,
    _biased_grouped_topk_baseline,
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


def _aiter_moe_act_quant(activation, swiglu_limit):
    if swiglu_limit is not None:
        return None
    return "mxfp8" if activation == "situv2" else "mxfp4"


def _fused_moe_baseline(
    original,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation="silu",
    act_quant=None,
    w1_scale=None,
    w2_scale=None,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    gate_mode="separated",
    use_shuffled_weight=False,
    **_,
):
    from aiter import ActivationType, QuantType

    if act_quant != _aiter_moe_act_quant(activation, swiglu_limit):
        return None  # aiter picks the activation quant by its own rules
    if swiglu_limit is not None and swiglu_limit < 10:
        return None  # aiter's CK-Tile decode path does not clamp
    if activation == "gelu" and expert_mask is not None:
        return None  # aiter has no gelu a4w4 under expert parallelism
    w1.is_shuffled = w2.is_shuffled = use_shuffled_weight

    def call():
        return original(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            activation=getattr(ActivationType, activation.capitalize()),
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            num_local_tokens=num_local_tokens,
            swiglu_limit=swiglu_limit,
            beta=beta,
            linear_beta=linear_beta,
            gate_mode=gate_mode,
        )

    if activation != "situv2":
        return call

    def situv2_call():
        # aiter reads SiTUv2's mxfp8 activations (act_quant) from its environment at call time
        before = os.environ.get("AITER_SITUV2_A8W4")
        os.environ["AITER_SITUV2_A8W4"] = "1"
        try:
            return call()
        finally:
            if before is None:
                del os.environ["AITER_SITUV2_A8W4"]
            else:
                os.environ["AITER_SITUV2_A8W4"] = before

    return situv2_call


@adapter("aiter.fused_moe.fused_moe", fused_moe, _fused_moe_baseline)
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
        "act_quant": _aiter_moe_act_quant(activation, swiglu_limit),
        "expert_mask": expert_mask,
        "beta": beta,
        "linear_beta": linear_beta,
        "swiglu_limit": swiglu_limit,
        "num_local_tokens": num_local_tokens,
        "gate_mode": getattr(gate_mode, "value", gate_mode) or "separated",
        "use_shuffled_weight": getattr(w1, "is_shuffled", False) is True,
    }


def _gemm_a16w16_baseline(original, a, b, bias=None, out_dtype=None, use_shuffled_weight=False, **_):
    b.is_shuffled = use_shuffled_weight
    return lambda: original(A=a, B=b, bias=bias, otype=out_dtype)


@adapter("aiter.tuned_gemm.gemm_a16w16", gemm_bf16_nt, _gemm_a16w16_baseline)
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


def _gemm_a8w8_baseline(
    original,
    a,
    b,
    a_scale,
    b_scale,
    bias=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    if use_shuffled_weight:
        return None
    return lambda: original(XQ=a, WQ=b, x_scale=a_scale, w_scale=b_scale, bias=bias, dtype=out_dtype)


@adapter("aiter.ops.gemm_op_a8w8.gemm_a8w8", gemm_a8w8_nt, _gemm_a8w8_baseline)
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
def _gemm_a8w8_blockscale_baseline(
    original,
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    if scale_major_mode != "K" or use_shuffled_weight:
        return None
    return lambda: original(XQ=a, WQ=b, x_scale=a_scale, w_scale=b_scale, dtype=out_dtype)


@adapter(
    "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale",
    gemm_fp8_nt_blockscaled,
    _gemm_a8w8_blockscale_baseline,
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


def _triton_gemm_a8w8_blockscale_baseline(
    original,
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    if scale_major_mode != "K" or use_shuffled_weight:
        return None
    return lambda: original(x=a, w=b, x_scale=a_scale, w_scale=b_scale, dtype=out_dtype, y=out)


@adapter(
    "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale.gemm_a8w8_blockscale",
    gemm_fp8_nt_blockscaled,
    _triton_gemm_a8w8_blockscale_baseline,
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


def _gemm_a8w8_blockscale_bpreshuffle_baseline(
    original,
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    if not use_shuffled_weight:
        return None
    if scale_major_mode == "MN":
        return lambda: original(XQ=a, WQ=b, x_scale=a_scale.mT, w_scale=b_scale, dtype=out_dtype, out=out)
    # a K-major a_scale is relaid out in the timed call: aiter takes the MN-major one
    return lambda: original(
        XQ=a,
        WQ=b,
        x_scale=a_scale.mT.contiguous().mT,
        w_scale=b_scale,
        dtype=out_dtype,
        out=out,
    )


@adapter(
    "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle",
    gemm_fp8_nt_blockscaled,
    _gemm_a8w8_blockscale_bpreshuffle_baseline,
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


def _gemm_a4w4_baseline(
    original,
    a,
    b,
    a_descale,
    b_descale,
    bias=None,
    out_dtype=None,
    use_shuffled_weight=False,
    **_,
):
    if not use_shuffled_weight:
        return None
    return lambda: original(A=a, B=b, A_scale=a_descale, B_scale=b_descale, bias=bias, dtype=out_dtype)


@adapter("aiter.ops.gemm_op_a4w4.gemm_a4w4", gemm_mxfp4_nt, _gemm_a4w4_baseline)
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


def _batched_gemm_bf16_baseline(original, A, B, out_dtype=None, use_shuffled_weight=False, **_):
    if use_shuffled_weight:
        return None
    return lambda: original(XQ=A, WQ=B, dtype=out_dtype)


@adapter(
    "aiter.ops.batched_gemm_op_bf16.batched_gemm_bf16_CK",
    bmm_bf16_nt,
    _batched_gemm_bf16_baseline,
)
def _(XQ, WQ, bias=None, dtype=None, splitK=None):
    # splitK is scheduling; the registry decides.
    if bias is not None:
        return None
    if XQ.dtype != torch.bfloat16:
        return None
    return {"A": XQ, "B": WQ, "out_dtype": dtype}


def _mhc_pre_baseline(
    original,
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
    **_,
):
    return lambda: original(
        residual=residual,
        fn=proj_weight,
        hc_scale=mix_scale,
        hc_base=mix_bias,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


@adapter("aiter.ops.mhc.mhc_pre", mhc_pre, _mhc_pre_baseline)
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


def _mhc_post_baseline(original, x, residual, post_mix, comb_mix, out=None, **_):
    if out is None:
        return None
    return lambda: original(
        out=out,
        x=x,
        residual=residual,
        post_layer_mix=post_mix.flatten(1),
        comb_res_mix=comb_mix,
    )


@adapter("aiter.ops.mhc.mhc_post", mhc_post, _mhc_post_baseline)
def _(out, x, residual, post_layer_mix, comb_res_mix, store_nt=-1):
    # store_nt is a store/cache strategy, not semantics.
    return {
        "x": x,
        "residual": residual,
        "post_mix": post_layer_mix.view(*residual.shape[:2], 1),
        "comb_mix": comb_res_mix,
        "out": out,
    }


def _mhc_fused_post_pre_baseline(
    original,
    x,
    residual,
    post_mix,
    comb_mix,
    proj_weight,
    mix_scale,
    mix_bias,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
    **_,
):
    return lambda: original(
        layer_input=x,
        residual_in=residual,
        post_layer_mix=post_mix.flatten(1),
        comb_res_mix=comb_mix,
        fn=proj_weight,
        hc_scale=mix_scale,
        hc_base=mix_bias,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


@adapter(
    "aiter.ops.mhc.mhc_fused_post_pre",
    mhc_fused_post_pre,
    _mhc_fused_post_pre_baseline,
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


def _rmsnorm2d_fwd_with_add_baseline(original, x, residual, weight, epsilon, normalized, residual_out, **_):
    return lambda: original(
        out=normalized,
        input=x,
        residual_in=residual,
        residual_out=residual_out,
        weight=weight,
        epsilon=epsilon,
    )


@adapter(
    "aiter.ops.rmsnorm.rmsnorm2d_fwd_with_add",
    add_rms_norm,
    _rmsnorm2d_fwd_with_add_baseline,
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


def _rmsnorm2d_fwd_baseline(original, x, weight, epsilon, **_):
    return lambda: original(x, weight, epsilon)


@adapter("aiter.ops.rmsnorm.rmsnorm2d_fwd", rms_norm, _rmsnorm2d_fwd_baseline)
def _(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
    if use_model_sensitive_rmsnorm:
        return None
    return {"x": input, "weight": weight, "epsilon": epsilon}


def _deepgemm_fp8_paged_mqa_logits_baseline(
    original,
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    page_size,
    logits=None,
    preshuffled=True,
    **_,
):
    if logits is None:
        return None
    return lambda: original(
        q_fp8=q,
        kv_cache=kv_cache,
        weights=weights,
        out_logits=logits,
        context_lens=context_lens,
        kv_indices=block_tables,
        max_model_len=logits.shape[1],
        Preshuffle=preshuffled,
        KVBlockSize=page_size,
    )


@adapter(
    "aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits",
    fp8_paged_mqa_logits,
    _deepgemm_fp8_paged_mqa_logits_baseline,
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
def _per_group_quant_hip_baseline(original, x, **_):
    return lambda: original(x, group_size=128, quant_dtype=torch.float8_e4m3fn, transpose_scale=True)


@adapter("aiter.ops.quant.per_group_quant_hip", per_group_quant_fp8, _per_group_quant_hip_baseline)
def _(*_, **__):
    return None


def _per_1x32_mx_quant_hip_baseline(original, x, **_):
    from aiter import dtypes

    return lambda: original(x, quant_dtype=torch.float8_e4m3fn, scale_type=dtypes.fp8_e8m0, shuffle=False)


@adapter("aiter.ops.quant.per_1x32_mx_quant_hip", per_group_quant_mxfp8, _per_1x32_mx_quant_hip_baseline)
def _(*_, **__):
    return None


def _pa_sparse_prefill_fp8_opus_baseline(
    original,
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    softmax_scale=512**-0.5,
    output=None,
    **_,
):
    return lambda: original(
        q_nope.view(torch.float8_e4m3fn),
        q_rope,
        prefix_nope.view(torch.float8_e4m3fn),
        prefix_rope,
        prefix_indices,
        prefix_indptr,
        extend_nope.view(torch.float8_e4m3fn),
        extend_rope,
        extend_indices,
        extend_indptr,
        sinks,
        softmax_scale,
        out=output,
    )


@adapter(
    "aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus",
    sparse_prefill_mxfp8,
    _pa_sparse_prefill_fp8_opus_baseline,
)
def _(*_, **__):
    return None
