# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's Triton GEMM, ``aiter.ops.triton.gemm_a16w16.gemm_a16w16``, served by AKLR's ``gemm`` op."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm

__all__ = ["gemm_a16w16"]


def gemm_a16w16(x, w, *, bias, dtype, y, activation, skip_reduce):
    """AKLR's ``x @ w.T + bias`` into ``y``, or why aiter computes it. aiter's config, kernel type,
    backend and persistence pick its schedule, not the result, and are not taken."""
    if activation is not None or skip_reduce:
        return FallbackReason.OUT_OF_CONTRACT
    return gemm(x, w, bias=bias, out_dtype=dtype, out=y)
