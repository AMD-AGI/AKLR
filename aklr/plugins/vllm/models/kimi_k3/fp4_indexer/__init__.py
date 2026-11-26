# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's DSA indexer prefill with an MXFP4 index cache (vLLM's ``models/kimi_k3/amd/ops/fp4_indexer_prefill.py``)."""

from aklr.plugins.vllm.models.kimi_k3.fp4_indexer.api import kimi_k3_fp4_indexer_prefill

__all__ = ["kimi_k3_fp4_indexer_prefill"]
