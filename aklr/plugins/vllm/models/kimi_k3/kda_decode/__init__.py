# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA speculative decode: each sequence's window of drafts to verify (vLLM's ``fused_recurrent_kda``)."""

from aklr.plugins.vllm.models.kimi_k3.kda_decode.api import kimi_k3_kda_decode_spec

__all__ = ["kimi_k3_kda_decode_spec"]
