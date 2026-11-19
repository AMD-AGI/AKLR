# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1 on ROCm (vLLM's ``models/deepseek_v41/amd``), which vLLM runs uncompiled; its
ops are the sglang plugin's (``aklr.plugins.sglang.models.deepseekv41``)."""

from aklr.plugins.vllm.models.deepseekv41 import api

__all__ = ["api"]
