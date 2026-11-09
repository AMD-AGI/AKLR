# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from vLLM's layers to the ops of ``aklr/ops``; mirrors vLLM's
``model_executor/layers``."""

from aklr.plugins.vllm.layers import fused_moe, gemm

__all__ = ["fused_moe", "gemm"]
