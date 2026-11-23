# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import kimi_k3_latent_moe, kimi_k3_mla_prefill_gate, kimi_k3_mla_vc_output_gate

__all__ = ["kimi_k3_latent_moe", "kimi_k3_mla_prefill_gate", "kimi_k3_mla_vc_output_gate"]
