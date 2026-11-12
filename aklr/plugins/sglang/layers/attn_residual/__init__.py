# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention-residual aggregation, as sglang runs it on ROCm."""

from aklr.plugins.sglang.layers.attn_residual.api import add_attn_res, attn_res

__all__ = ["add_attn_res", "attn_res"]
