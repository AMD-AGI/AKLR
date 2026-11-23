# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import deepseek_v2_fp8_moe, deepseek_v2_mxfp4_moe

__all__ = ["deepseek_v2_fp8_moe", "deepseek_v2_mxfp4_moe"]
