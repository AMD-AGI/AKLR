# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["rms_norm_add_reference", "rms_norm_reference"]


def _normed(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    import torch

    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """x times the reciprocal rms of its row, times weight."""
    x = to_float64(row_major_args["x"], point.dtypes["x"]).float()
    weight = to_float64(row_major_args["weight"], point.dtypes["weight"]).float()
    return _normed(x, weight, row_major_args["eps"]).double()


def rms_norm_add_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(the norm of x + residual, x + residual)."""
    summed = (
        to_float64(row_major_args["x"], point.dtypes["x"]).float()
        + to_float64(row_major_args["residual"], point.dtypes["residual"]).float()
    )
    weight = to_float64(row_major_args["weight"], point.dtypes["weight"]).float()
    return _normed(summed, weight, row_major_args["eps"]).double(), summed.double()
