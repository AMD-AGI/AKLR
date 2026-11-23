# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import qwen2_moe_fp8_experts, qwen2_moe_fp8_router_experts, qwen2_moe_mxfp4_moe, qwen2_moe_router

__all__ = ["qwen2_moe_fp8_experts", "qwen2_moe_fp8_router_experts", "qwen2_moe_mxfp4_moe", "qwen2_moe_router"]
