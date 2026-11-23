# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import DEEPSEEK_V2_FP8_MOE, DEEPSEEK_V2_MXFP4_MOE, family

CONTRACTS = family.contracts

__all__ = ["CONTRACTS", "DEEPSEEK_V2_FP8_MOE", "DEEPSEEK_V2_MXFP4_MOE", "family"]
