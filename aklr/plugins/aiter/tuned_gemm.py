# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's unquantized GEMM, ``aiter.tuned_gemm.tgemm.mm``, served by AKLR's ``gemm`` op."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm

__all__ = ["mm"]


def mm(inp, weights, *, bias, otype, scale_a, scale_b, scale_c, out=None):
    """AKLR's ``inp @ weights.T + bias``, or why aiter computes it; ``out`` is the caller's
    buffer, which aiter takes from 0.1.28 on."""
    if scale_a is not None or scale_b is not None or scale_c is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return gemm(inp, weights, bias=bias, out_dtype=otype, out=out)
