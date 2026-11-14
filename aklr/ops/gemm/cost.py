# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one GEMM call, for each contract of this op (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Mapping

from aklr.core.contract.cost import Cost


def gemm_cost(symbols: Mapping[str, int | str | bool], tensor_bytes: Mapping[str, int]) -> Cost:
    """``2 * batch * m * n * k`` operations at the rate of ``a``'s dtype; every operand,
    scale and output moved once."""
    flops = 2 * int(symbols.get("batch", 1)) * int(symbols["m"]) * int(symbols["n"]) * int(symbols["k"])
    return Cost(flops, sum(tensor_bytes.values()), str(symbols["a.dtype"]))


# every contract of this op does a gemm's work
gemm_a8w8_cost = gemm_a8w8_blockwise_scaled_cost = gemm_a4w4_mxfp4_cost = bmm_cost = gemm_cost
