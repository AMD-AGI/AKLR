# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import (
    QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT,
    QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT,
    QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT,
    family,
)

CONTRACTS = family.contracts

__all__ = [
    "CONTRACTS",
    "QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT",
    "QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT",
    "QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT",
    "family",
]
