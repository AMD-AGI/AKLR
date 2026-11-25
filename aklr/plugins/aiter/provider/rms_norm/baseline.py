# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = ["RmsnormBaseline", "RmsnormWithAddBaseline"]


class RmsnormWithAddBaseline(OpBaseline):
    def baseline(
        self,
        *,
        input,
        residual_in,
        weight,
        epsilon=1e-6,
        gemma_norm=False,
        use_model_sensitive_rmsnorm=0,
        out=None,
        residual_out=None,
    ):
        """aiter's fused add + RMSNorm, into the outputs; both are ``input``'s shape and dtype."""
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add

        if out is None:
            out = torch.empty_like(input)
        if residual_out is None:
            residual_out = torch.empty_like(input)
        rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)
        return {"out": out, "residual_out": residual_out}


class RmsnormBaseline(OpBaseline):
    def baseline(self, *, input, weight, epsilon=1e-6, use_model_sensitive_rmsnorm=0, normalized=None):
        """aiter's RMSNorm."""
        from aiter.ops.rmsnorm import rmsnorm2d_fwd

        return {"normalized": rmsnorm2d_fwd(input, weight, epsilon)}
