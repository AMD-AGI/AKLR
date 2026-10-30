# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route, must_default
from aklr.ops.rms_norm.contract import ADD_RMS_NORM

__all__ = ["add_rms_norm"]


@api(ADD_RMS_NORM)
def add_rms_norm(x, residual, weight, epsilon, normalized, residual_out): ...


@aklr_route("aiter.ops.rmsnorm.rmsnorm2d_fwd_with_add")
def _(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    gemma_norm=False,
    use_model_sensitive_rmsnorm=0,
):
    must_default(gemma_norm, use_model_sensitive_rmsnorm)
    add_rms_norm(
        x=input,
        residual=residual_in,
        weight=weight,
        epsilon=epsilon,
        normalized=out,
        residual_out=residual_out,
    )
