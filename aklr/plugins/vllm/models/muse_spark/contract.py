# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import MUSE_SPARK_MOE, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "MUSE_SPARK_MOE", "family"]
