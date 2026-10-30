# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route, must_default, unused
from aklr.ops.mhc.contract import MHC_FUSED_POST_PRE, MHC_POST, MHC_PRE

__all__ = ["mhc_fused_post_pre", "mhc_post", "mhc_pre"]


@api(MHC_PRE)
def mhc_pre(
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
): ...


@api(MHC_POST)
def mhc_post(x, residual, post_mix, comb_mix, *, out=None): ...


@api(MHC_FUSED_POST_PRE)
def mhc_fused_post_pre(
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
): ...


@aklr_route("aiter.ops.mhc.mhc_pre")
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
    unused(large_m_splitk)  # Scheduling, not part of the mathematical contract.
    must_default(is_fn_pack_bf16)  # Packed projection storage is not yet supported.
    return mhc_pre(
        residual=residual,
        proj_weight=fn,
        mix_scale=hc_scale,
        mix_bias=hc_base,
        rms_eps=rms_eps,
        pre_eps=hc_pre_eps,
        sinkhorn_eps=hc_sinkhorn_eps,
        post_multiplier=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


@aklr_route("aiter.ops.mhc.mhc_post")
def _(out, x, residual, post_layer_mix, comb_res_mix, store_nt=-1):
    unused(store_nt)  # Store/cache strategy, not mathematical semantics.
    mhc_post(
        x=x,
        residual=residual,
        post_mix=post_layer_mix,
        comb_mix=comb_res_mix,
        out=out,
    )


@aklr_route("aiter.ops.mhc.mhc_fused_post_pre")
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
    unused(force_fused)  # Scheduling, not part of the mathematical contract.
    must_default(is_fn_pack_bf16)  # Packed projection storage is not yet supported.
    return mhc_fused_post_pre(
        x=layer_input,  # aiter's input; the contract's layer_input is an output
        residual=residual_in,
        post_mix=post_layer_mix,
        comb_mix=comb_res_mix,
        proj_weight=fn,
        mix_scale=hc_scale,
        mix_bias=hc_base,
        rms_eps=rms_eps,
        pre_eps=hc_pre_eps,
        sinkhorn_eps=hc_sinkhorn_eps,
        post_multiplier=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
