# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = [
    "deepseek_v4_hc_boundary_norm_baseline",
    "deepseek_v4_hc_boundary_norm_mxfp8_baseline",
    "deepseek_v4_hc_pre_norm_baseline",
]


_HC_MULT = 4


def _hc_boundary(
    x, residual, post_in, comb_in, pre_prev, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
):
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import hc_boundary_fused_deferred

    return hc_boundary_fused_deferred(
        x,
        residual,
        post_in,
        comb_in,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        _HC_MULT,
        sinkhorn_iters,
        rms_eps,
        hc_eps,
    )


def deepseek_v4_hc_boundary_norm_baseline(
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
    residual_out=None,
    norm=None,
    pre=None,
    post=None,
    comb=None,
):
    """What sglang 0.5.23 runs on gfx950 for an FFN boundary of
    forward_hc_pre_from_prev_fused_boundary: hc_boundary_fused_deferred (the boundary's partials),
    then rmsnorm_with_sinkhorn without the fp8 operand (the bf16 norm launch hosting the reduce +
    Sinkhorn); the outputs in the contract's order."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn

    new_residual, y, coefficients = _hc_boundary(
        x, residual, post_in, comb_in, pre_prev, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
    )
    _, out = rmsnorm_with_sinkhorn(y, norm_weight, norm_eps, coefficients, fake_quant=False)
    return (new_residual, out, *coefficients.tensors())


def deepseek_v4_hc_boundary_norm_mxfp8_baseline(
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
    residual_out=None,
    norm=None,
    quantized=None,
    scales=None,
    pre=None,
    post=None,
    comb=None,
):
    """The same boundary with the norm launch also emitting its MXFP8 operand, as sglang 0.5.23
    runs the attention boundary for an MXFP8 wqkv_a (input_norm_fake_quant) and the FFN boundary
    under SGLANG_HIP_FFN_NORM_MXFP8; the outputs in the contract's order."""
    import torch
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn

    new_residual, y, coefficients = _hc_boundary(
        x, residual, post_in, comb_in, pre_prev, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
    )
    operand, out = rmsnorm_with_sinkhorn(y, norm_weight, norm_eps, coefficients, emit_fp8=True)
    # sglang holds the E8M0 bytes as uint8
    scale = operand.scale.view(torch.float8_e8m0fnu)
    return (new_residual, out, operand.q, scale, *coefficients.tensors())


def deepseek_v4_hc_pre_norm_baseline(
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
    norm=None,
    pre=None,
    post=None,
    comb=None,
):
    """What sglang 0.5.23 runs on gfx950 (SGLANG_USE_AITER) for the attention input of a DSpark
    draft stage (NonCudaDSparkMixin._forward_hc_pre_from_prev): mhc_combine and the aiter RMSNorm
    (combine), then hc_mix_stats_sinkhorn (mix_stats); the outputs in the contract's order."""
    from aiter import rmsnorm2d_fwd
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn, mhc_combine

    flat = residual.flatten(1)
    out = rmsnorm2d_fwd(mhc_combine(flat, pre_prev, _HC_MULT, residual.dtype), norm_weight, norm_eps)
    mixes = hc_mix_stats_sinkhorn(flat, hc_fn, hc_scale, hc_base, _HC_MULT, sinkhorn_iters, rms_eps, hc_eps)
    return (out, *mixes)
