# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_rms_norm_reference"]


def add_rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(normalized, residual_out) from the exact sum x + residual."""
    total = to_float64(row_major_args["x"], point.dtypes["x"]) + to_float64(
        row_major_args["residual"], point.dtypes["residual"]
    )
    weight = to_float64(row_major_args["weight"], point.dtypes["weight"])
    mean_square = total.square().mean(dim=-1, keepdim=True)
    normalized = total * (mean_square + row_major_args["epsilon"]).rsqrt() * weight
    return normalized, total
