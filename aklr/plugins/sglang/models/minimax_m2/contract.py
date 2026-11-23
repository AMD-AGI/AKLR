# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import MINIMAX_M2_FP8_MOE, MINIMAX_M2_MXFP4_MOE, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "MINIMAX_M2_FP8_MOE", "MINIMAX_M2_MXFP4_MOE", "family"]
