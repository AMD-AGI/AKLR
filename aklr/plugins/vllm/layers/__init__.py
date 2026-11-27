# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from vLLM's layers to aiter's functions; mirrors vLLM's
``model_executor/layers`` and ``v1/attention``."""

from aklr.plugins.vllm.layers import dcp, fused_moe_router, utils

__all__ = ["dcp", "fused_moe_router", "utils"]
