# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from vLLM's model files to AKLR ops; mirrors vLLM's ``models``. A model whose forward
vLLM runs without torch.compile (``@ignore_torch_compile``) runs its Python at every eager
call and CUDA graph capture, so a route can replace any function of it."""

from aklr.plugins.vllm.models import muse_spark

__all__ = ["muse_spark"]
