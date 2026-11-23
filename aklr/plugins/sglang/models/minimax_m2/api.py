# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import minimax_m2_fp8_moe, minimax_m2_mxfp4_moe

__all__ = ["minimax_m2_fp8_moe", "minimax_m2_mxfp4_moe"]
