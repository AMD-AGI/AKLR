###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the mhc op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations


def mhc_pre_baseline(
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
    from aiter.ops.mhc import mhc_pre

    return lambda: mhc_pre(
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


def mhc_post_baseline(x, residual, post_mix, comb_mix, out=None, **_):
    from aiter.ops.mhc import mhc_post

    if out is None:
        return None
    return lambda: mhc_post(
        out=out,
        x=x,
        residual=residual,
        post_layer_mix=post_mix.flatten(1),
        comb_res_mix=comb_mix,
    )


def mhc_fused_post_pre_baseline(
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
    from aiter.ops.mhc import mhc_fused_post_pre

    return lambda: mhc_fused_post_pre(
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
