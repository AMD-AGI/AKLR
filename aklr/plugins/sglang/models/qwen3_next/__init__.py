# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next Gated DeltaNet prefill and decode operations."""

from aklr.plugins.sglang.models.qwen3_next.api import (
    qwen3_next_gdn_decode_group_fp8_quant,
    qwen3_next_gdn_prefill_group_fp8_quant,
)

__all__ = ["qwen3_next_gdn_decode_group_fp8_quant", "qwen3_next_gdn_prefill_group_fp8_quant"]
