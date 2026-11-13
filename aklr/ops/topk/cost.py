# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one biased_grouped_topk call (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Mapping

from aklr.core.contract.cost import Cost


def biased_grouped_topk_cost(
    symbols: Mapping[str, int | str | bool], tensor_bytes: Mapping[str, int]
) -> Cost:
    """A router top-k moves each tensor once and does no matrix math."""
    return Cost.from_tensor_bytes(tensor_bytes)
