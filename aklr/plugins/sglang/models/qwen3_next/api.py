# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import (
    qwen3_next_gated_attention_decode_group_fp8_quant,
    qwen3_next_gdn_decode_group_fp8_quant,
    qwen3_next_gdn_prefill_group_fp8_quant,
)

__all__ = [
    "qwen3_next_gated_attention_decode_group_fp8_quant",
    "qwen3_next_gdn_decode_group_fp8_quant",
    "qwen3_next_gdn_prefill_group_fp8_quant",
]
