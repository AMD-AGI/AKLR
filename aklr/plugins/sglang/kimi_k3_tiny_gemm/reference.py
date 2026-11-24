# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_tiny_gemm_reference"]


def kimi_k3_tiny_gemm_reference(point: Any, row_major_args: Mapping[str, Any]) -> Tensor:
    """``x @ w.T``."""
    x = to_float32(row_major_args["x"], point.dtypes["x"])
    return x @ to_float32(row_major_args["w"], point.dtypes["w"]).mT
