# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's RMSNorm ops, under aiter's own names and parameters."""

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.plugins.aiter.provider.rms_norm.contract import RMSNORM, RMSNORM_WITH_ADD

__all__ = ["rmsnorm", "rmsnorm_with_add"]


@api(RMSNORM)
def rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm=0): ...


@api(RMSNORM_WITH_ADD)
def rmsnorm_with_add(
    out, input, residual_in, residual_out, weight, epsilon, gemma_norm=False, use_model_sensitive_rmsnorm=0
): ...
