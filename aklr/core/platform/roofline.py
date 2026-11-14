# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A GPU's roofline: the least time a call's work (``Cost``) takes on it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aklr.core.contract.cost import Cost

__all__ = ["Roofline"]


@dataclass(frozen=True)
class Roofline:
    """The ceilings of one GPU; the defaults are MI355X's, measured on it (aklr-kb)."""

    # matrix-core FLOP/s by operand dtype. Measured: bf16 2.44-2.47e15 and fp8 4.90-4.95e15 (triton/attention.md);
    # fp16 as bf16, int8 as fp8 and fp4 at twice fp8 follow the spec's ratios, unmeasured
    flops_per_second: Mapping[str, float] = field(
        default_factory=lambda: {
            "bfloat16": 2.45e15,
            "float16": 2.45e15,
            "float8_e4m3fn": 4.92e15,
            "float8_e5m2": 4.92e15,
            "int8": 4.92e15,
            "float4_e2m1": 9.84e15,
        }
    )
    bytes_per_second: float = 7.1e12  # a streaming HBM read, 7.13-7.17e12 measured (triton/attention.md)
    # AKLR's graph timer reads no call faster: the fastest of 3017 kernel times in main's full bench of
    # 2026-11-13 (ROCm 7.15) was 4.18 µs; pitfalls/aklr_dispatch_floor.md saw 4.40-4.56 on older stacks
    floor_seconds: float = 4.1e-6

    def seconds(self, cost: Cost) -> float:
        """The least time ``cost`` takes: its math at its dtype's rate or its bytes at the
        bandwidth, whichever is longer, never under the timer's floor. ``ValueError`` for
        math in a dtype without a rate."""
        math_seconds = 0.0
        if cost.flops:
            if cost.dtype not in self.flops_per_second:
                raise ValueError(f"no matrix-core rate for {cost.dtype}")
            math_seconds = cost.flops / self.flops_per_second[cost.dtype]
        return max(math_seconds, cost.bytes / self.bytes_per_second, self.floor_seconds)
