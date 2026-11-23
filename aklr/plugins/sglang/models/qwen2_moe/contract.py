# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import (
    QWEN2_MOE_FP8_EXPERTS,
    QWEN2_MOE_FP8_ROUTER_EXPERTS,
    QWEN2_MOE_MXFP4_MOE,
    QWEN2_MOE_ROUTER,
    family,
)

CONTRACTS = family.contracts

__all__ = [
    "CONTRACTS",
    "QWEN2_MOE_FP8_EXPERTS",
    "QWEN2_MOE_FP8_ROUTER_EXPERTS",
    "QWEN2_MOE_MXFP4_MOE",
    "QWEN2_MOE_ROUTER",
    "family",
]
