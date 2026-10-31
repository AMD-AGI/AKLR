# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_rms_norm_reference"]


def add_rms_norm_reference(x, residual, weight, epsilon=1e-6) -> tuple[Tensor, Tensor]:
    """(normalized, residual_out) from the exact sum x + residual."""
    total = x.double() + residual.double()
    mean_square = total.square().mean(dim=-1, keepdim=True)
    return (mean_square + epsilon).rsqrt() * total * weight.double(), total
