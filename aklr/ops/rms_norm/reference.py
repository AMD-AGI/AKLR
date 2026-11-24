# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.reference import OpReference

__all__ = ["AddRmsNormReference", "RmsNormReference"]


class AddRmsNormReference(OpReference):
    def reference(self, *, x, residual, weight, epsilon, **_):
        """normalized and residual_out from the sum x + residual."""
        total = x.float() + residual.float()
        return {
            "normalized": _normalized(total, weight.float(), epsilon).to(x.dtype),
            "residual_out": total.to(x.dtype),
        }


class RmsNormReference(OpReference):
    def reference(self, *, x, weight, epsilon, **_):
        """x times the reciprocal rms of its row, times weight."""
        return {"normalized": _normalized(x.float(), weight.float(), epsilon).to(x.dtype)}


def _normalized(x, weight, epsilon):
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * (mean_square + epsilon).rsqrt() * weight
