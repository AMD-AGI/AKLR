# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one quant call (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Mapping

from aklr.core.contract.cost import Cost


def per_group_quant_fp8_cost(
    symbols: Mapping[str, int | str | bool], tensor_bytes: Mapping[str, int]
) -> Cost:
    """A per-group quantization moves each tensor once and does no matrix math."""
    return Cost.from_tensor_bytes(tensor_bytes)


per_group_quant_mxfp8_cost = per_group_quant_fp8_cost
