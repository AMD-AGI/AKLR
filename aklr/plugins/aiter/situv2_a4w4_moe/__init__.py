# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused MoE for SiTUv2 experts with MXFP4 activations, the a4w4 path vLLM's Kimi-K3 recipe takes."""

from aklr.plugins.aiter.situv2_a4w4_moe import api

__all__ = ["api"]
