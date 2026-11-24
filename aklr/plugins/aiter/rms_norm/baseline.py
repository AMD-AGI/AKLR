# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = ["AddRmsNormBaseline", "RmsNormBaseline"]


class AddRmsNormBaseline(OpBaseline):
    def baseline(self, *, x, residual, weight, epsilon=1e-6, normalized=None, residual_out=None):
        """aiter's fused add + RMSNorm, into the outputs; both are ``x``'s shape and dtype."""
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add

        if normalized is None:
            normalized = torch.empty_like(x)
        if residual_out is None:
            residual_out = torch.empty_like(x)
        rmsnorm2d_fwd_with_add(normalized, x, residual, residual_out, weight, epsilon)
        return {"normalized": normalized, "residual_out": residual_out}


class RmsNormBaseline(OpBaseline):
    def baseline(self, *, x, weight, epsilon=1e-6, normalized=None):
        """aiter's RMSNorm."""
        from aiter.ops.rmsnorm import rmsnorm2d_fwd

        return {"normalized": rmsnorm2d_fwd(x, weight, epsilon)}
