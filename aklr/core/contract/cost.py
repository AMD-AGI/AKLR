# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work one op call does, as a roofline reads it: an op's ``cost.py`` writes it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

__all__ = ["Cost"]


class Cost(NamedTuple):
    """The least work of one call: matrix-core operations, and bytes between memory and chip."""

    flops: int  # multiply-adds count two
    bytes: int  # each tensor the call reads or writes, moved once
    # the operand dtype whose matrix-core rate bounds ``flops``; None: no matrix math
    dtype: str | None

    @classmethod
    def from_tensor_bytes(cls, tensor_bytes: Mapping[str, int]) -> Cost:
        """A call that moves every tensor once and does no matrix math."""
        return cls(0, sum(tensor_bytes.values()), None)
