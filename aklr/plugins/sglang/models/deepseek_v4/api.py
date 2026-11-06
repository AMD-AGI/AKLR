# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4.contract import (
    DEEPSEEK_V4_HC_BOUNDARY_NORM,
    DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8,
    DEEPSEEK_V4_HC_PRE_NORM,
)

__all__ = ["deepseek_v4_hc_boundary_norm", "deepseek_v4_hc_boundary_norm_mxfp8", "deepseek_v4_hc_pre_norm"]


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
    rms_eps=1e-6,
    norm_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
): ...


@api(DEEPSEEK_V4_HC_PRE_NORM)
def deepseek_v4_hc_pre_norm(
    residual,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    rms_eps=1e-6,
    norm_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
): ...


class _HcMixes:
    """A boundary's pre / post / comb as the op computed them, in place of sglang's HcCoefficients
    for the readers the route leaves on the served path: the other boundary's op call or sglang's
    hc_boundary (pre, post, comb) and the layer's return (tensors()). The readers that need more
    (materialize, materialized), the norms hosting the reduce + Sinkhorn and the all-reduce mHC
    fusions, never see it: the norms only follow sglang's own boundary, and the route falls back
    whenever the fusions could run."""

    def __init__(self, pre, post, comb):
        self._mixes = (pre, post, comb)

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


def _boundary_args(layer, x, residual, mixes_in, hc_fn, hc_scale, hc_base, norm):
    """The op arguments of one boundary of a target decoder layer; mixes_in is the previous
    boundary's (pre, post, comb), whose post and comb only the deferred hc_post of x reads."""
    pre_prev, post_in, comb_in = mixes_in
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
    """The attention boundary and input_layernorm as one op call where sglang would run
    hc_boundary_fused_deferred, then rmsnorm_with_sinkhorn emitting wqkv_a's MXFP8 operand
    (deepseek_v4_hip.input_norm -> input_norm_fake_quant): (residual, norm, x_quant, mixes), or
    a FallbackReason."""
    import torch
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    x, residual, post, comb = pending_post
    # no deferred hc_post (the first layer, or one after a layer that could not defer), or the
    # other branches of input_norm: the plain norm, or the fp8-grid operand of a wqkv_a that
    # takes no MXFP8
    if x is None or prev_pre is None or not layer.fused_rmsnorm_fake_quant:
        return FallbackReason.OUT_OF_CONTRACT
    if not layer._wqkv_a_native_consumer_checked:
        # resolved on first use exactly as input_norm_fake_quant resolves it
        layer._wqkv_a_native_consumer = gfx95_dense._mxfp8_consumer(
            layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
        )
        layer._wqkv_a_native_consumer_checked = True
    if not layer._wqkv_a_native_consumer:
        return FallbackReason.OUT_OF_CONTRACT
    norm = layer.input_layernorm
    out = deepseek_v4_hc_boundary_norm_mxfp8(
        **_boundary_args(
            layer,
            x,
            residual,
            (prev_pre, post, comb),
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
    x_quant = Mxfp8Activation(quantized, scales.view(torch.uint8))
    return new_residual, normed, x_quant, _HcMixes(pre, post, comb)


def _ffn_input(layer, x, residual, attn_mixes):
    """The FFN boundary and post_attention_layernorm as one op call where sglang would run
    hc_boundary_fused_deferred, then rmsnorm_with_sinkhorn (deepseek_v4_gfx95_dense
    .post_attention_norm), with the shared expert's MXFP8 operand under
    SGLANG_HIP_FFN_NORM_MXFP8: (residual, norm, mixes), or a FallbackReason."""
    import torch
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc as fused_mhc
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    # off gfx950 the norm and the reduce + Sinkhorn run apart
    if not fused_mhc._is_gfx95_supported:
        return FallbackReason.OUT_OF_CONTRACT
    args = _boundary_args(
        layer,
        x,
        residual,
        attn_mixes.tensors(),
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        layer.post_attention_layernorm,
    )
    if not gfx95_dense._ffn_norm_emits_mxfp8(layer, x.shape[0]):
        out = deepseek_v4_hc_boundary_norm(**args)
        if isinstance(out, FallbackReason):
            return out
        new_residual, normed, pre, post, comb = out
        return new_residual, normed, _HcMixes(pre, post, comb)
    out = deepseek_v4_hc_boundary_norm_mxfp8(**args)
    if isinstance(out, FallbackReason):
        return out
    new_residual, normed, quantized, scales, pre, post, comb = out
    # the shared expert's gate_up takes it in place of its own quant (_forward_shared_experts)
    normed._hip_mxfp8_operand = Mxfp8Activation(quantized, scales.view(torch.uint8))
    return new_residual, normed, _HcMixes(pre, post, comb)


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
    """forward_hc_pre_from_prev_fused_boundary with each mHC boundary and the RMSNorm after it
    as one op call; a boundary the ops do not take runs sglang's boundary and norm, and the
    attention and the MoE stay sglang's. Where sglang could fuse a boundary's hc_post into the
    attention or MoE all-reduce (attention_mhc_fusion, moe_mhc_fusion: TP4 decode of V4.1), the
    whole layer stays sglang's."""
    from sglang.srt.models import deepseek_v4_platform_stage as stage
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc as fused_mhc

    if pending_post is None:
        pending_post = (None, hidden_states, None, None)
    # both fusions require _can_fuse_mhc of the boundary's new streams, which have the incoming
    # streams' shape, dtype and contiguity (sglang allocates them empty_like, the ops only take
    # contiguous ones)
    if fused_mhc._can_fuse_mhc(layer, pending_post[1], forward_batch):
        return FallbackReason.OUT_OF_CONTRACT
    attention = _attention_input(layer, pending_post, prev_pre)
    if isinstance(attention, FallbackReason):
        residual, x, attn_mixes = stage.make_amd_hc_boundary(
            layer,
            *pending_post,
            prev_pre,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
        )
        x, x_quant = layer._input_norm(x, allow_aiter_quant=False, coefficients=attn_mixes)
    else:
        residual, x, x_quant, attn_mixes = attention
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        x = layer.self_attn(
            x=x, positions=positions, forward_batch=forward_batch, x_quant=x_quant, defer_all_reduce=False
        )
    ffn = _ffn_input(layer, x, residual, attn_mixes)
    if isinstance(ffn, FallbackReason):
        residual, x, ffn_mixes = stage.make_amd_hc_boundary(
            layer,
            x,
            residual,
            attn_mixes.post,
            attn_mixes.comb,
            attn_mixes.pre,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
        )
        x = stage._gfx95_dense_post_attention_norm(layer, x, ffn_mixes)
    else:
        residual, x, ffn_mixes = ffn
    x = layer._run_moe_ffn_dp_sync(
        x, forward_batch, input_ids=input_ids, input_ids_global=input_ids_global, return_moe_output=False
    )
    ffn_pre, ffn_post, ffn_comb = ffn_mixes.tensors()
    if defer_post:
        return None, ffn_pre, (x, residual, ffn_post, ffn_comb)
    return layer.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre, None


def _draft_args(hc, residual, pre_prev):
    """The op arguments one hyper-connection of a DSpark draft stage reads (as mhc.combine and
    mhc.mix_stats read them)."""
    cfg = hc.cfg
    return {
        "residual": residual,
        "pre_prev": pre_prev,
        "hc_fn": hc.fn,
        "hc_scale": hc.scale,
        "hc_base": hc.base,
        "norm_weight": hc.norm.weight.data,
        "rms_eps": cfg.rms_eps,
        "norm_eps": hc.norm.variance_epsilon,
        "hc_eps": cfg.eps,
        "sinkhorn_iters": cfg.sinkhorn_iters,
    }


def _draft_norm_differs(norm) -> bool:
    """Whether the hyper-connection's RMSNorm module computes other than the contracts' norm."""
    return norm.cast_x_before_out_mul or norm.variance_size_override is not None


@sglang_route(
    "sglang.srt.models.deepseek_v4_platform_stage.NonCudaDSparkMixin._forward_hc_pre_from_prev",
    versions=("0.5.23",),
)
def _dspark_forward_hc_pre_from_prev(self, positions, hidden_states, forward_batch, prev_pre):
    """A DSpark draft stage with its two mHC boundaries as op calls: the attention input
    (mhc.combine and mhc.mix_stats of the stage's streams) as deepseek_v4_hc_pre_norm, and the
    attention's hc_post with the FFN input and mixes as deepseek_v4_hc_boundary_norm, the same op
    as the target's FFN boundary. The stage's last hc_post, the attention and the FFN stay
    sglang's; a boundary the ops do not take runs sglang's steps."""
    from sglang.srt.models import deepseek_v4_mhc as mhc

    assert self.attn_hc is not None and self.ffn_hc is not None
    # the CUDA stats stream overlaps mix_stats with the attention and the FFN
    if self.hc_stats_stream is not None and mhc.use_stats_stream(self.hc_cfg, forward_batch, hidden_states):
        return FallbackReason.OUT_OF_CONTRACT
    attention = FallbackReason.OUT_OF_CONTRACT
    # stage 0 collapses without a pre mix: its input is stream 0
    if prev_pre is not None and not _draft_norm_differs(self.attn_hc.norm):
        attention = deepseek_v4_hc_pre_norm(**_draft_args(self.attn_hc, hidden_states, prev_pre))
    if isinstance(attention, FallbackReason):
        x = mhc.combine(self.attn_hc, mhc.HcState(hidden_states, prev_pre))
        attn_pre, attn_post, attn_comb = mhc.mix_stats(self.attn_hc, hidden_states)
    else:
        x, attn_pre, attn_post, attn_comb = attention
    with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
        x = self.self_attn(positions, x, forward_batch)
    ffn = FallbackReason.OUT_OF_CONTRACT
    if not _draft_norm_differs(self.ffn_hc.norm):
        ffn = deepseek_v4_hc_boundary_norm(
            x=x,
            post_in=attn_post,
            comb_in=attn_comb,
            **_draft_args(self.ffn_hc, hidden_states, attn_pre),
        )
    if isinstance(ffn, FallbackReason):
        residual = self.hc_post(x, hidden_states, attn_post, attn_comb)
        x = mhc.combine(self.ffn_hc, mhc.HcState(residual, attn_pre))
        ffn_pre, ffn_post, ffn_comb = mhc.mix_stats(self.ffn_hc, residual)
    else:
        residual, x, ffn_pre, ffn_post, ffn_comb = ffn
    x = self._run_ffn(x, forward_batch)
    return self.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre
