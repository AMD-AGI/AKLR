# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA prefill and decode as sglang's Triton KDA backend runs them
(``srt/layers/attention/linear/kernels/kda_triton.py``)."""

from aklr.plugins.sglang.layers.kda_triton.api import (
    kimi_k3_sgl_kda_decode,
    kimi_k3_sgl_kda_prefill,
    kimi_k3_sgl_kda_prefill_track,
)

__all__ = ["kimi_k3_sgl_kda_decode", "kimi_k3_sgl_kda_prefill", "kimi_k3_sgl_kda_prefill_track"]
