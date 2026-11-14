# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one rms_norm call (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Mapping

from aklr.core.contract.cost import Cost


def rms_norm_cost(symbols: Mapping[str, int | str | bool], tensor_bytes: Mapping[str, int]) -> Cost:
    """An RMS norm moves each tensor once and does no matrix math."""
    return Cost.from_tensor_bytes(tensor_bytes)


add_rms_norm_cost = rms_norm_cost
