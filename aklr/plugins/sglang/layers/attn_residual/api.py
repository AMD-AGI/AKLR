# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import add_attn_res, attn_res
from .plugin import declare as declare

__all__ = ["add_attn_res", "attn_res"]
