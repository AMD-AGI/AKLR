# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA prefill short convolution (aiter's causal conv1d with the q/k/v split, as vLLM's ``kda.py`` calls it)."""

from aklr.ops.kda_conv.api import kimi_k3_kda_conv

__all__ = ["kimi_k3_kda_conv"]
