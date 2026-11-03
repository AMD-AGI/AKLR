# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["add_rms_norm_baseline", "rms_norm_baseline"]


def add_rms_norm_baseline(x, residual, weight, epsilon=1e-6, normalized=None, residual_out=None):
    """aiter's fused add + RMSNorm, into the outputs; both are ``x``'s shape and dtype."""
    import torch
    from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add

    if normalized is None:
        normalized = torch.empty_like(x)
    if residual_out is None:
        residual_out = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(normalized, x, residual, residual_out, weight, epsilon)
    return normalized, residual_out


def rms_norm_baseline(x, weight, epsilon=1e-6):
    """aiter's RMSNorm."""
    from aiter.ops.rmsnorm import rmsnorm2d_fwd

    return rmsnorm2d_fwd(x, weight, epsilon)
