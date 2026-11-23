# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports from this family's plugin module."""

from .plugin import KIMI_K3_LATENT_MOE, KIMI_K3_MLA_PREFILL_GATE, KIMI_K3_MLA_VC_OUTPUT_GATE, family

CONTRACTS = family.contracts

__all__ = [
    "CONTRACTS",
    "KIMI_K3_LATENT_MOE",
    "KIMI_K3_MLA_PREFILL_GATE",
    "KIMI_K3_MLA_VC_OUTPUT_GATE",
    "family",
]
