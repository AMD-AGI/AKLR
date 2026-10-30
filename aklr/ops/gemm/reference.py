# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "bmm_reference",
    "gemm_a4w4_mxfp4_reference",
    "gemm_a8w8_blockwise_scaled_reference",
    "gemm_a8w8_reference",
    "gemm_reference",
]


def _decoded(
    point: Any,
    row_major_args: Mapping[str, Any],
    name: str,
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor | None:
    t = row_major_args.get(name)
    return None if t is None else to_float64(t, point.dtypes[name])


def _weight(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    b = _decoded(point, row_major_args, "b", to_float64)
    return b if point.attrs["trans_b"] else b.mT


def _product(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    a = _decoded(point, row_major_args, "a", to_float64)
    return a @ _weight(point, row_major_args, to_float64).mT


def _out(product: Tensor, bias: Tensor | None) -> Tensor:
    return product if bias is None else product + bias


def gemm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """``a @ b.T + bias``."""
    return _out(
        _product(point, row_major_args, to_float64),
        _decoded(point, row_major_args, "bias", to_float64),
    )


def gemm_a8w8_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Per-row scales; a squeezed ``(rows,)`` scale reads as ``(rows, 1)``."""
    a_scale = _decoded(point, row_major_args, "a_scale", to_float64)
    b_scale = _decoded(point, row_major_args, "b_scale", to_float64)
    a = _decoded(point, row_major_args, "a", to_float64) * a_scale.reshape(-1, 1)
    b = _weight(point, row_major_args, to_float64) * b_scale.reshape(-1, 1)
    return _out(a @ b.mT, _decoded(point, row_major_args, "bias", to_float64))


def gemm_a8w8_blockwise_scaled_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """A scale per ``(1, 128)`` of ``a`` and ``(128, 128)`` of ``b``; tails partial."""
    a = _decoded(point, row_major_args, "a", to_float64)
    b = _weight(point, row_major_args, to_float64)
    (n, k), block = b.shape, 128
    a_scale = _decoded(point, row_major_args, "a_scale", to_float64)
    if point.attrs["scale_major_mode"] == "MN":
        a_scale = a_scale.mT
    a_scale = a_scale.repeat_interleave(block, dim=1)
    b_scale = _decoded(point, row_major_args, "b_scale", to_float64)
    b_scale = b_scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    out = (a * a_scale[:, :k]) @ (b * b_scale[:n, :k]).mT
    return _out(out, _decoded(point, row_major_args, "bias", to_float64))


def gemm_a4w4_mxfp4_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """``alpha * (a @ b.T) + bias``, one e8m0 scale per 32 fp4 values along ``k``."""

    def scaled(x: str) -> Tensor:
        scale = _decoded(point, row_major_args, f"{x}_scale", to_float64)
        decoded = _decoded(point, row_major_args, x, to_float64)
        return decoded * scale.repeat_interleave(32, dim=-1)

    return _out(
        row_major_args["alpha"] * (scaled("a") @ scaled("b").mT),
        _decoded(point, row_major_args, "bias", to_float64),
    )


def bmm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """``bias`` is ``(batch, n)``, broadcast over ``m``."""
    bias = _decoded(point, row_major_args, "bias", to_float64)
    return _out(
        _product(point, row_major_args, to_float64),
        None if bias is None else bias.unsqueeze(-2),
    )
