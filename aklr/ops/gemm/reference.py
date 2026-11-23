# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

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
) -> Tensor | None:
    t = row_major_args.get(name)
    return None if t is None else to_float32(t, point.dtypes[name])


def _weight(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    b = _decoded(point, row_major_args, "b")
    return b if point.attrs["trans_b"] else b.mT


def _product(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    a = _decoded(point, row_major_args, "a")
    return a @ _weight(point, row_major_args).mT


def _out(product: Tensor, bias: Tensor | None) -> Tensor:
    return product if bias is None else product + bias


def gemm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """``a @ b.T + bias``."""
    return _out(
        _product(point, row_major_args),
        _decoded(point, row_major_args, "bias"),
    )


def _int8_product(a: Tensor, b: Tensor) -> Tensor:
    """a b^T of int8 values, summed exactly: a float32 sum of 1024 products (each at most 2**14
    in size) is an exact integer, and so is the int32 sum of those while k < 2**17."""
    import torch

    k = a.shape[-1]
    parts = ((a[:, s : s + 1024] @ b[:, s : s + 1024].mT).to(torch.int32) for s in range(0, k, 1024))
    return sum(parts).float()


def gemm_a8w8_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """Per-row scales; a squeezed ``(rows,)`` scale reads as ``(rows, 1)``. int8 operands sum
    exactly in int32, then take the scales."""
    a_scale = _decoded(point, row_major_args, "a_scale").reshape(-1, 1)
    b_scale = _decoded(point, row_major_args, "b_scale").reshape(1, -1)
    a = _decoded(point, row_major_args, "a")
    b = _weight(point, row_major_args)
    if point.dtypes["a"] == "int8":
        product = _int8_product(a, b) * a_scale * b_scale
    else:
        product = (a * a_scale) @ (b * b_scale.mT).mT
    return _out(product, _decoded(point, row_major_args, "bias"))


def gemm_a8w8_blockwise_scaled_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """A scale per ``(1, 128)`` of ``a`` and ``(128, 128)`` of ``b``; tails partial."""
    a = _decoded(point, row_major_args, "a")
    b = _weight(point, row_major_args)
    (n, k), block = b.shape, 128
    a_scale = _decoded(point, row_major_args, "a_scale")
    if point.attrs["scale_major_mode"] == "MN":
        a_scale = a_scale.mT
    a_scale = a_scale.repeat_interleave(block, dim=1)
    b_scale = _decoded(point, row_major_args, "b_scale")
    b_scale = b_scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    out = (a * a_scale[:, :k]) @ (b * b_scale[:n, :k]).mT
    return _out(out, _decoded(point, row_major_args, "bias"))


def gemm_a4w4_mxfp4_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """``alpha * (a @ b.T) + bias``, one e8m0 scale per 32 fp4 values along ``k``."""

    def scaled(x: str) -> Tensor:
        scale = _decoded(point, row_major_args, f"{x}_scale")
        decoded = _decoded(point, row_major_args, x)
        return decoded * scale.repeat_interleave(32, dim=-1)

    return _out(
        row_major_args["alpha"] * (scaled("a") @ scaled("b").mT),
        _decoded(point, row_major_args, "bias"),
    )


def bmm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """``bias`` is ``(batch, n)``, broadcast over ``m``."""
    bias = _decoded(point, row_major_args, "bias")
    return _out(
        _product(point, row_major_args),
        None if bias is None else bias.unsqueeze(-2),
    )
