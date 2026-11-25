# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.reference import OpReference

__all__ = ["RmsnormReference", "RmsnormWithAddReference"]


class RmsnormWithAddReference(OpReference):
    def reference(self, *, input, residual_in, weight, epsilon, **_):
        """out and residual_out from the sum input + residual_in."""
        total = input.float() + residual_in.float()
        return {
            "out": _normalized(total, weight.float(), epsilon).to(input.dtype),
            "residual_out": total.to(input.dtype),
        }


class RmsnormReference(OpReference):
    def reference(self, *, input, weight, epsilon, **_):
        """input times the reciprocal rms of its row, times weight."""
        return {"normalized": _normalized(input.float(), weight.float(), epsilon).to(input.dtype)}


def _normalized(x, weight, epsilon):
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * (mean_square + epsilon).rsqrt() * weight
