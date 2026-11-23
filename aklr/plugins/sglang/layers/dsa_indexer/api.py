# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import dsa_indexer_key_prepare, dsa_indexer_prepare

__all__ = ["dsa_indexer_key_prepare", "dsa_indexer_prepare"]
