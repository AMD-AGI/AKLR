# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's RMSNorm ops ``rmsnorm`` and ``rmsnorm_with_add``: AKLR's contracts of those names take
aiter's calls as they are."""

from __future__ import annotations

from aklr.plugins.aiter.provider._base import offer, selector, served
from aklr.plugins.aiter.provider.rms_norm.api import rmsnorm, rmsnorm_with_add


@selector("rmsnorm", version=2)
def _select_rmsnorm(args, kwargs):
    return offer(rmsnorm.serving(*args), _run_rmsnorm)


def _run_rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm):
    return served(rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm))


@selector("rmsnorm_with_add")
def _select_rmsnorm_with_add(args, kwargs):
    return offer(rmsnorm_with_add.serving(*args), _run_rmsnorm_with_add)


def _run_rmsnorm_with_add(
    out, input, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm
):
    served(
        rmsnorm_with_add(
            out, input, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm
        )
    )
