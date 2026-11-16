# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work one op call does, as a roofline reads it (``OpContract.cost_at``)."""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["Cost"]


class Cost(NamedTuple):
    """The least work of one call: matrix-core operations, and bytes between memory and chip."""

    flops: int  # multiply-adds count two
    bytes_moved: int  # each tensor the call reads or writes, moved once
    # the operand dtype whose matrix-core rate bounds ``flops``; None: no matrix math
    flops_dtype: str | None
    # False: the contract has no formula in aklr/ops/cost.py and is not known to be memory-bound,
    # so its matrix math, if any, is uncounted and its roof may be far too low
    exact: bool = True
