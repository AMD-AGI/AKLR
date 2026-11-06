# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4.contract import (
    DEEPSEEK_V4_HC_BOUNDARY_NORM,
    DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8,
)

__all__ = ["deepseek_v4_hc_boundary_norm", "deepseek_v4_hc_boundary_norm_mxfp8"]


@api(DEEPSEEK_V4_HC_BOUNDARY_NORM)
def deepseek_v4_hc_boundary_norm(
    x,
    residual,
    post_in,
    comb_in,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps=1e-6,
    norm_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
): ...


@api(DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8)
def deepseek_v4_hc_boundary_norm_mxfp8(
    x,
    residual,
    post_in,
    comb_in,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps=1e-6,
    norm_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
): ...


class _Mixes:
    """A boundary's pre / post / comb, already computed: what sglang's HcCoefficients holds once
    its reduce + Sinkhorn has run."""

    materialized = True

    def __init__(self, pre, post, comb):
        self._mixes = (pre, post, comb)
        self.num_rows = pre.shape[0]

    def materialize(self) -> None:
        return None

    @property
    def pre(self):
        return self._mixes[0]

    @property
    def post(self):
        return self._mixes[1]

    @property
    def comb(self):
        return self._mixes[2]

    def tensors(self):
        return self._mixes


def _boundary_args(layer, x, residual, post_in, comb_in, pre_prev, hc_fn, hc_scale, hc_base, norm):
    return {
        "x": x,
        "residual": residual,
        "post_in": post_in,
        "comb_in": comb_in,
        "pre_prev": pre_prev,
        "hc_fn": hc_fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "norm_weight": norm.weight.data,
        "rms_eps": layer.rms_norm_eps,
        "norm_eps": norm.variance_epsilon,
        "hc_eps": layer.hc_eps,
        "sinkhorn_iters": layer.hc_sinkhorn_iters,
    }


def _attention_input(layer, pending_post, prev_pre):
    """The attention boundary and input_norm as one op call, where sglang's input_norm would run
    rmsnorm_with_sinkhorn emitting the MXFP8 operand of wqkv_a: (residual, norm, x_quant,
    coefficients), or a FallbackReason."""
    import torch
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    x, residual, post, comb = pending_post
    norm = layer.input_layernorm
    # the other branches of deepseek_v4_hip.input_norm, and input_norm_fake_quant's fp8-grid operand
    if x is None or prev_pre is None or layer.hc_mult != 4 or not layer.fused_rmsnorm_fake_quant:
        return FallbackReason.OUT_OF_CONTRACT
    if not layer._wqkv_a_native_consumer_checked:
        layer._wqkv_a_native_consumer = gfx95_dense._mxfp8_consumer(
            layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
        )
        layer._wqkv_a_native_consumer_checked = True
    if not layer._wqkv_a_native_consumer:
        return FallbackReason.OUT_OF_CONTRACT
    out = deepseek_v4_hc_boundary_norm_mxfp8(
        **_boundary_args(
            layer,
            x,
            residual,
            post,
            comb,
            prev_pre,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            norm,
        )
    )
    if isinstance(out, FallbackReason):
        return out
    new_residual, normed, quantized, scales, pre, post, comb = out
    # sglang's consumers read the E8M0 scales as uint8 bytes
    return new_residual, normed, Mxfp8Activation(quantized, scales.view(torch.uint8)), _Mixes(pre, post, comb)


def _ffn_input(layer, x, residual, attn_coefficients):
    """The FFN boundary and post_attention_norm as one op call, where sglang's post_attention_norm
    would run rmsnorm_with_sinkhorn: (residual, norm, coefficients), or a FallbackReason."""
    import torch
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc as fused_mhc
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    if x is None or layer.hc_mult != 4 or not fused_mhc._is_gfx95_supported:
        return FallbackReason.OUT_OF_CONTRACT
    norm = layer.post_attention_layernorm
    args = _boundary_args(
        layer,
        x,
        residual,
        attn_coefficients.post,
        attn_coefficients.comb,
        attn_coefficients.pre,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        norm,
    )
    if not gfx95_dense._ffn_norm_emits_mxfp8(layer, x.shape[0]):
        out = deepseek_v4_hc_boundary_norm(**args)
        if isinstance(out, FallbackReason):
            return out
        new_residual, normed, pre, post, comb = out
        return new_residual, normed, _Mixes(pre, post, comb)
    out = deepseek_v4_hc_boundary_norm_mxfp8(**args)
    if isinstance(out, FallbackReason):
        return out
    new_residual, normed, quantized, scales, pre, post, comb = out
    # the shared expert's gate_up takes it in place of its own quant (_forward_shared_experts)
    normed._hip_mxfp8_operand = Mxfp8Activation(quantized, scales.view(torch.uint8))
    return new_residual, normed, _Mixes(pre, post, comb)


@sglang_route(
    "sglang.srt.models.deepseek_v4_platform_stage.forward_hc_pre_from_prev_fused_boundary",
    versions=("0.5.23",),
)
def _forward_hc_pre_from_prev_fused_boundary(
    layer,
    positions,
    hidden_states,
    input_ids,
    forward_batch,
    input_ids_global,
    prev_pre,
    pending_post,
    defer_post,
):
    """forward_hc_pre_from_prev_fused_boundary with each mHC boundary and the RMSNorm after it as
    one op call; a boundary the ops do not take runs sglang's boundary and norm."""
    from sglang.srt.models import deepseek_v4_platform_stage as stage

    if pending_post is None:
        pending_post = (None, hidden_states, None, None)
    attention = _attention_input(layer, pending_post, prev_pre)
    if isinstance(attention, FallbackReason):
        residual, x, attn_coefficients = stage.make_amd_hc_boundary(
            layer,
            *pending_post,
            prev_pre,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
        )
        x, x_quant = layer._input_norm(x, allow_aiter_quant=False, coefficients=attn_coefficients)
    else:
        residual, x, x_quant, attn_coefficients = attention
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        mhc = stage.attention_mhc_fusion(layer, residual, attn_coefficients, forward_batch)
        x = layer.self_attn(
            x=x,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=x_quant,
            defer_all_reduce=mhc is not None,
        )
    if mhc is not None:
        assert isinstance(x, stage.AttnOutput)
        residual = stage.apply_attention_mhc(x.partial, mhc)
        x = None
    ffn = _ffn_input(layer, x, residual, attn_coefficients)
    if isinstance(ffn, FallbackReason):
        residual, x, ffn_coefficients = stage.make_amd_hc_boundary(
            layer,
            x,
            residual,
            attn_coefficients.post if x is not None else None,
            attn_coefficients.comb if x is not None else None,
            attn_coefficients.pre,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
        )
        x = stage._gfx95_dense_post_attention_norm(layer, x, ffn_coefficients)
    else:
        residual, x, ffn_coefficients = ffn
    mhc = stage.moe_mhc_fusion(layer, residual, ffn_coefficients, forward_batch)
    x = layer._run_moe_ffn_dp_sync(
        x,
        forward_batch,
        input_ids=input_ids,
        input_ids_global=input_ids_global,
        return_moe_output=mhc is not None,
    )
    ffn_pre, ffn_post, ffn_comb = ffn_coefficients.tensors()
    if mhc is not None:
        assert isinstance(x, stage.MoEOutput)
        return stage.apply_moe_mhc(x.get_merged(), mhc), ffn_pre, None
    if defer_post:
        return None, ffn_pre, (x, residual, ffn_post, ffn_comb)
    return layer.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre, None
