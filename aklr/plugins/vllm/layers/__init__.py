# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from vLLM's layers to the ops of ``aklr/ops``; mirrors vLLM's
``model_executor/layers`` and ``v1/attention``."""

from aklr.plugins.vllm.layers import attention, fused_moe, gemm

__all__ = ["attention", "fused_moe", "gemm"]
