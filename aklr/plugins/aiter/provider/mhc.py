# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's mHC pre, ``mhc_pre``, served by AKLR's ``mhc_pre``."""

from __future__ import annotations

from aklr.ops.mhc.api import mhc_pre
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
