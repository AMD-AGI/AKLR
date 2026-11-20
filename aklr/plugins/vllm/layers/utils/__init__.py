# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's ``model_executor/layers/utils``: its unquantized GEMM on ROCm, where it reaches vLLM's
skinny GEMMs; the routes go to ``aklr.ops.gemm``."""

from aklr.plugins.vllm.layers.utils import api

__all__ = ["api"]
