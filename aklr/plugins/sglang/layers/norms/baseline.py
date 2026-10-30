# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's RMSNorm.forward_aiter runs."""

from __future__ import annotations

__all__ = ["rms_norm_add_baseline", "rms_norm_baseline"]


def rms_norm_baseline(x, weight, eps, out_dtype=None, out=None):
    from aiter import rmsnorm2d_fwd

    return rmsnorm2d_fwd(x, weight, eps)


def rms_norm_add_baseline(x, residual, weight, eps, normed=None, residual_out=None):
    import torch
    from aiter import rmsnorm2d_fwd_with_add

    out, summed = torch.empty_like(x), torch.empty_like(x)
    rmsnorm2d_fwd_with_add(out, x, residual, summed, weight, eps)
    return out, summed
