# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import HEAD_DIM, KDA_CHAIN_VERIFY_RING, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "HEAD_DIM", "KDA_CHAIN_VERIFY_RING", "family"]
