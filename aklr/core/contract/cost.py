# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work one op call does, as a roofline reads it: ``aklr/ops/cost.py`` writes each contract's."""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["Cost"]


class Cost(NamedTuple):
    """The least work of one call: matrix-core operations, and bytes between memory and chip."""

    flops: int  # multiply-adds count two
    bytes: int  # each tensor the call reads or writes, moved once
    # the operand dtype whose matrix-core rate bounds ``flops``; None: no matrix math
    dtype: str | None
