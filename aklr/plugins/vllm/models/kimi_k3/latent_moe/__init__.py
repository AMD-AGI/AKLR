# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's latent MoE under tensor parallelism: its tail (``ROCmLatentMoERunner._shard_up_proj_tail``) in one
kernel that does the TP reductions over peer memory."""

from aklr.plugins.vllm.models.kimi_k3.latent_moe.api import kimi_k3_latent_moe_tail

__all__ = ["kimi_k3_latent_moe_tail"]
