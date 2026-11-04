###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.mhc.schema import MHC_FUSED_POST_PRE, MHC_POST, MHC_PRE

__all__ = ["mhc_fused_post_pre", "mhc_post", "mhc_pre"]


@aklr_api(MHC_PRE)
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


@aklr_api(MHC_POST)
def mhc_post(x, residual, post_mix, comb_mix, *, out=None): ...


@aklr_api(MHC_FUSED_POST_PRE)
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
