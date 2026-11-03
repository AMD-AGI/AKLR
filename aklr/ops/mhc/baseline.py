# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["mhc_fused_post_pre_baseline", "mhc_post_baseline", "mhc_pre_baseline"]


def mhc_pre_baseline(
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
    post_mix=None,
    comb_mix=None,
    layer_input=None,
):
    """aiter's mHC pre; it returns the outputs in the contract's order."""
    from aiter.ops.mhc import mhc_pre

    return mhc_pre(
        residual,
        proj_weight,
        mix_scale,
        mix_bias,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def mhc_post_baseline(x, residual, post_mix, comb_mix, *, out_dtype=None, out=None):
    """aiter's mHC post, into ``out``; ``out`` is ``residual``'s shape and dtype."""
    import torch
    from aiter.ops.mhc import mhc_post

    if out is None:
        out = torch.empty_like(residual)
    mhc_post(out, x, residual, post_mix, comb_mix)
    return out


def mhc_fused_post_pre_baseline(
    x,
    residual,
    post_mix,
    comb_mix,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
    next_post_mix=None,
    next_comb_mix=None,
    layer_input=None,
    next_residual=None,
):
    """aiter's fused mHC post and pre; it returns the outputs in the contract's order."""
    from aiter.ops.mhc import mhc_fused_post_pre

    return mhc_fused_post_pre(
        x,  # aiter's layer_input is this input; the contract's is an output
        residual,
        post_mix,
        comb_mix,
        proj_weight,
        mix_scale,
        mix_bias,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
