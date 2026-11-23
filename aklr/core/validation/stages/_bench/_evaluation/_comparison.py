# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["find_nondeterminism"]


def find_nondeterminism(
    first_outputs: Mapping[str, Tensor], second_outputs: Mapping[str, Tensor]
) -> str | None:
    """``None`` when two runs of a ``bitwise`` schema on the same inputs wrote byte-identical
    mutable tensors, else the detail."""
    fraction = max(_differing_fraction(first_outputs[o], second_outputs[o]) for o in first_outputs)
    return f"{fraction:.3g} of an output's elements differ from numeric/0's" if fraction else None


def _differing_fraction(a: Tensor, b: Tensor) -> float:
    """The fraction of elements whose bytes differ between two runs (NaN equals itself)."""
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"runs differ in shape or dtype: {a.shape}, {b.shape}")
    if not a.numel():
        return 0.0
    a_bytes, b_bytes = (t.contiguous().reshape(-1).view(torch.uint8).reshape(t.numel(), -1) for t in (a, b))
    return int((a_bytes != b_bytes).any(-1).sum()) / a.numel()
