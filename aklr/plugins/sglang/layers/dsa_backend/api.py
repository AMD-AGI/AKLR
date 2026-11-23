# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import dsa_sparse_mla_decode, dsa_sparse_mla_prefill, dsa_topk_transform

__all__ = ["dsa_sparse_mla_decode", "dsa_sparse_mla_prefill", "dsa_topk_transform"]
