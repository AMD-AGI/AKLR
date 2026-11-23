# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import ADD_ATTN_RES, ATTN_RES, family

CONTRACTS = family.contracts

__all__ = ["ADD_ATTN_RES", "ATTN_RES", "CONTRACTS", "family"]
