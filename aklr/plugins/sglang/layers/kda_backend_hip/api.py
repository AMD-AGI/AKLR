# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import kda_chain_verify_ring

__all__ = ["kda_chain_verify_ring"]
