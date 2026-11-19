# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 on ROCm (vLLM's ``models/kimi_k3/amd``), whose forward vLLM runs uncompiled under
full CUDA graphs."""

from aklr.plugins.vllm.models.kimi_k3 import attn_res, kda_spec_stage_c, latent_moe

__all__ = ["attn_res", "kda_spec_stage_c", "latent_moe"]
