# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import DSA_SPARSE_MLA_DECODE, DSA_SPARSE_MLA_PREFILL, DSA_TOPK_TRANSFORM, ROPE_DIM, family

CONTRACTS = family.contracts

__all__ = [
    "CONTRACTS",
    "DSA_SPARSE_MLA_DECODE",
    "DSA_SPARSE_MLA_PREFILL",
    "DSA_TOPK_TRANSFORM",
    "ROPE_DIM",
    "family",
]
