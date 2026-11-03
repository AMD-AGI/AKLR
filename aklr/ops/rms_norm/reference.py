# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_rms_norm_reference", "rms_norm_reference"]


def add_rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(normalized, residual_out) from the exact sum x + residual."""
    args, dtypes = row_major_args, point.dtypes
    total = decode(args["x"], dtypes["x"]) + decode(args["residual"], dtypes["residual"])
    return _normalized(total, decode(args["weight"], dtypes["weight"]), args["epsilon"]), total


def rms_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """x times the reciprocal rms of its row, times weight."""
    args, dtypes = row_major_args, point.dtypes
    return _normalized(
        decode(args["x"], dtypes["x"]), decode(args["weight"], dtypes["weight"]), args["epsilon"]
    )


def _normalized(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    return x * (x.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt() * weight
