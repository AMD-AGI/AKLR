# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_rms_norm_reference", "rms_norm_reference"]


def add_rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """(normalized, residual_out) from the sum x + residual."""
    args, dtypes = row_major_args, point.dtypes
    total = to_float32(args["x"], dtypes["x"]) + to_float32(args["residual"], dtypes["residual"])
    return _normalized(total, to_float32(args["weight"], dtypes["weight"]), args["epsilon"]), total


def rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """x times the reciprocal rms of its row, times weight."""
    args, dtypes = row_major_args, point.dtypes
    return _normalized(
        to_float32(args["x"], dtypes["x"]), to_float32(args["weight"], dtypes["weight"]), args["epsilon"]
    )


def _normalized(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * (mean_square + epsilon).rsqrt() * weight
