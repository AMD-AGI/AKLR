# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import to_float32

__all__ = [
    "BmmReference",
    "GemmA4w4Mxfp4Reference",
    "GemmA8w8BlockwiseScaledReference",
    "GemmA8w8Reference",
    "GemmReference",
]


def _weight(b, trans_b):
    b = b.float()
    return b if trans_b else b.mT


def _out(product, bias):
    return product if bias is None else product + bias.float()


class GemmReference(OpReference):
    def reference(self, *, a, b, trans_b, out_dtype, bias=None, **_):
        """``a @ b.T + bias``."""
        return {"out": _out(a.float() @ _weight(b, trans_b).mT, bias).to(out_dtype)}


def _int8_product(a, b):
    """a b^T of int8 values, summed exactly: a float32 sum of 1024 products (each at most 2**14
    in size) is an exact integer, and so is the int32 sum of those while k < 2**17."""
    k = a.shape[-1]
    parts = ((a[:, s : s + 1024] @ b[:, s : s + 1024].mT).to(torch.int32) for s in range(0, k, 1024))
    return sum(parts).float()


class GemmA8w8Reference(OpReference):
    def reference(self, *, a, b, a_scale, b_scale, trans_b, out_dtype, bias=None, **_):
        """Per-row scales; a squeezed ``(rows,)`` scale reads as ``(rows, 1)``. int8 operands sum
        exactly in int32, then take the scales."""
        a_scale = a_scale.float().reshape(-1, 1)
        b_scale = b_scale.float().reshape(1, -1)
        is_int8 = a.dtype == torch.int8
        a, b = a.float(), _weight(b, trans_b)
        if is_int8:
            product = _int8_product(a, b) * a_scale * b_scale
        else:
            product = (a * a_scale) @ (b * b_scale.mT).mT
        return {"out": _out(product, bias).to(out_dtype)}


class GemmA8w8BlockwiseScaledReference(OpReference):
    def reference(self, *, a, b, a_scale, b_scale, trans_b, scale_major_mode, out_dtype, bias=None, **_):
        """A scale per ``(1, 128)`` of ``a`` and ``(128, 128)`` of ``b``; tails partial."""
        a, b = a.float(), _weight(b, trans_b)
        (n, k), block = b.shape, 128
        a_scale = a_scale.float()
        if scale_major_mode == "MN":
            a_scale = a_scale.mT
        a_scale = a_scale.repeat_interleave(block, dim=1)
        b_scale = b_scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
        out = (a * a_scale[:, :k]) @ (b * b_scale[:n, :k]).mT
        return {"out": _out(out, bias).to(out_dtype)}


def _mxfp4(x, scale):
    """An MXFP4 operand in float32, dequantized by its e8m0 scale per 32 values along k."""
    return to_float32(x, "float4_e2m1") * to_float32(scale, "float8_e8m0").repeat_interleave(32, dim=-1)


class GemmA4w4Mxfp4Reference(OpReference):
    def reference(self, *, a, b, a_scale, b_scale, alpha, out_dtype, bias=None, **_):
        """``alpha * (a @ b.T) + bias``, one e8m0 scale per 32 fp4 values along ``k``."""
        out = _out(alpha * (_mxfp4(a, a_scale) @ _mxfp4(b, b_scale).mT), bias)
        return {"out": out.to(out_dtype)}


class BmmReference(OpReference):
    def reference(self, *, a, b, trans_b, bias=None, **_):
        """``bias`` is ``(batch, n)``, broadcast over ``m``."""
        out = _out(a.float() @ _weight(b, trans_b).mT, None if bias is None else bias.unsqueeze(-2))
        return {"out": out.to(a.dtype)}
