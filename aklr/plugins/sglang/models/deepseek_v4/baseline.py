# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang runs on gfx950 for one boundary of forward_hc_pre_from_prev_fused_boundary (sglang
0.5.23): hc_boundary_fused_deferred (the boundary partials), then rmsnorm_with_sinkhorn (the norm
launch hosting the reduce + Sinkhorn)."""

from __future__ import annotations

__all__ = [
    "deepseek_v4_hc_boundary_norm_baseline",
    "deepseek_v4_hc_boundary_norm_mxfp8_baseline",
]

_STREAMS = 4


def _boundary(
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
        _STREAMS,
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
    *,
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
    """The boundary partials, then the plain bf16 norm hosting the reduce + Sinkhorn (the FFN
    boundary without the shared expert's MXFP8 operand); the outputs in the contract's order."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn

    new_residual, y, coefficients = _boundary(
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
    *,
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
    """The boundary partials, then the norm emitting its MXFP8 operand and hosting the reduce +
    Sinkhorn; the outputs in the contract's order."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn

    new_residual, y, coefficients = _boundary(
        x, residual, post_in, comb_in, pre_prev, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
    )
    operand, out = rmsnorm_with_sinkhorn(y, norm_weight, norm_eps, coefficients, emit_fp8=True)
    import torch

    # sglang holds the E8M0 bytes as uint8
    return (new_residual, out, operand.q, operand.scale.view(torch.float8_e8m0fnu), *coefficients.tensors())
