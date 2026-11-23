# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's mHC pre, ``mhc_pre``, served by AKLR's ``mhc_pre``, and its post fused with the next pre,
``mhc_fused_post_pre``, by AKLR's ``mhc_fused_post_pre``."""

from __future__ import annotations

from aklr.ops.mhc.api import mhc_fused_post_pre, mhc_pre
from aklr.plugins.aiter.provider._base import offer, selector, served


def _kwargs(rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps):
    return {
        "rms_eps": rms_eps,
        "pre_eps": hc_pre_eps,
        "sinkhorn_eps": hc_sinkhorn_eps,
        "post_multiplier": hc_post_mult_value,
        "sinkhorn_iters": sinkhorn_repeat,
        "norm_weight": norm_weight,
        "norm_eps": norm_eps,
    }


@selector("mhc_pre")
def _select_mhc_pre(args, kwargs):
    residual, fn, hc_scale, hc_base, *scalars, _large_m_splitk, w_preshuffle_bf16, res_preshuffle = args
    # the shuffled fn or residual layouts are not AKLR's; large_m_splitk only picks aiter's schedule
    if w_preshuffle_bf16 or res_preshuffle:
        return None
    return offer(mhc_pre.serving(residual, fn, hc_scale, hc_base, **_kwargs(*scalars)), _run_mhc_pre)


def _run_mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
    large_m_splitk,
    w_preshuffle_bf16,
    res_preshuffle,
):
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    # (post_mix, comb_mix, layer_input), aiter's order
    return served(mhc_pre(residual, fn, hc_scale, hc_base, **scalars))


def _post_mix(post_layer_mix):
    """aiter's post mix, (T, 4, 1) or (T, 4), as AKLR's (T, 4); ``None`` for another shape."""
    if post_layer_mix.dim() == 3 and post_layer_mix.shape[-1] == 1:
        return post_layer_mix.squeeze(-1)
    return post_layer_mix if post_layer_mix.dim() == 2 else None


@selector("mhc_fused_post_pre")
def _select_mhc_fused_post_pre(args, kwargs):
    (
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        w_preshuffle_bf16,
        res_preshuffle,
    ) = args
    post_mix = _post_mix(post_layer_mix)
    # the shuffled fn or residual layouts are not AKLR's
    if w_preshuffle_bf16 or res_preshuffle or post_mix is None:
        return None
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    return offer(
        mhc_fused_post_pre.serving(
            layer_input, residual_in, post_mix, comb_res_mix, fn, hc_scale, hc_base, **scalars
        ),
        _run_mhc_fused_post_pre,
    )


def _run_mhc_fused_post_pre(
    layer_input,
    residual_in,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
    w_preshuffle_bf16,
    res_preshuffle,
):
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    # (post_mix, comb_mix, layer_input, next_residual), aiter's order
    return served(
        mhc_fused_post_pre(
            layer_input,
            residual_in,
            _post_mix(post_layer_mix),
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            **scalars,
        )
    )
