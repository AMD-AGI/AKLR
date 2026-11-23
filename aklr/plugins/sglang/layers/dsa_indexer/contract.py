# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import (
    DSA_INDEXER_KEY_PREPARE,
    DSA_INDEXER_PREPARE,
    FP8_MAX,
    HEAD_DIM,
    PAGE_SIZE,
    ROPE_DIM,
    SCALE_FLOOR,
    family,
)

CONTRACTS = family.contracts

__all__ = [
    "CONTRACTS",
    "DSA_INDEXER_KEY_PREPARE",
    "DSA_INDEXER_PREPARE",
    "FP8_MAX",
    "HEAD_DIM",
    "PAGE_SIZE",
    "ROPE_DIM",
    "SCALE_FLOOR",
    "family",
]
