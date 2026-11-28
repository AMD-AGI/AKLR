# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA as sglang's Triton KDA backend runs it: the prefill
(``srt/layers/attention/linear/kernels/kda_triton.py``) and, on sglang 0.5.23, the chain verify in
ReplaySSM ring mode (``srt/layers/attention/linear/kda_backend.py``)."""

from aklr.plugins.sglang.layers.kda_triton.api import (
    kimi_k3_sgl_kda_prefill,
    kimi_k3_sgl_kda_prefill_track,
    kimi_k3_sgl_kda_verify,
)

__all__ = ["kimi_k3_sgl_kda_prefill", "kimi_k3_sgl_kda_prefill_track", "kimi_k3_sgl_kda_verify"]
