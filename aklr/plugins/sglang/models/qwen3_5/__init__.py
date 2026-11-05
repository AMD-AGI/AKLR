# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI
# Applies to the GDN operations.
"""Qwen3.5 Gated DeltaNet prefill and decode operations, and the full-attention layers'
gated attention decode."""

from aklr.plugins.sglang.models.qwen3_5.api import (
    gated_attention_decode_group_fp8_quant,
    gdn_decode,
    gdn_decode_group_fp8_quant,
    gdn_prefill_group_fp8_quant,
)

__all__ = [
    "gated_attention_decode_group_fp8_quant",
    "gdn_decode",
    "gdn_decode_group_fp8_quant",
    "gdn_prefill_group_fp8_quant",
]
