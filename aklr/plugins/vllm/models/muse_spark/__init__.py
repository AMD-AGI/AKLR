# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark on ROCm (vLLM's ``models/muse_spark/amd``), which vLLM runs uncompiled under
breakable CUDA graphs."""

from aklr.plugins.vllm.models.muse_spark import api

__all__ = ["api"]
