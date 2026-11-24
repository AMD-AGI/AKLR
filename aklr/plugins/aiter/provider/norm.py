# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's RMSNorm ops served by AKLR's ``rms_norm`` and ``add_rms_norm``."""

from __future__ import annotations

from aklr.plugins.aiter.provider._base import offer, selector, served
from aklr.plugins.aiter.rms_norm.api import add_rms_norm, rms_norm


@selector("rmsnorm", version=2)
def _select_rmsnorm(args, kwargs):
    x, weight, epsilon, use_model_sensitive_rmsnorm = args
    if use_model_sensitive_rmsnorm:
        return None
    return offer(rms_norm.serving(x, weight, epsilon), _run_rmsnorm)


def _run_rmsnorm(x, weight, epsilon, use_model_sensitive_rmsnorm):
    return served(rms_norm(x, weight, epsilon))


@selector("rmsnorm_with_add")
def _select_rmsnorm_with_add(args, kwargs):
    out, x, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm = args
    if gemma_norm or use_model_sensitive_rmsnorm:
        return None
    return offer(
        add_rms_norm.serving(x, residual_in, weight, epsilon, out, residual_out),
        _run_rmsnorm_with_add,
    )


def _run_rmsnorm_with_add(
    out, x, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm
):
    served(add_rms_norm(x, residual_in, weight, epsilon, out, residual_out))
