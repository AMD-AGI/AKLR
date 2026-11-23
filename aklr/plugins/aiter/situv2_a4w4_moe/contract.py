# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import SITUV2_A4W4_MOE, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "SITUV2_A4W4_MOE", "family"]
