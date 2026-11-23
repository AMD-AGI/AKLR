# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import KIMI_K3_ATTN_RES, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "KIMI_K3_ATTN_RES", "family"]
