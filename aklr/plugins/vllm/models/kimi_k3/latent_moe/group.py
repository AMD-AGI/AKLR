# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["kimi_k3_latent_moe_tail_group_init"]


def kimi_k3_latent_moe_tail_group_init() -> None:
    """The kernel's state over the group's process group (one rank a device), as the vLLM route sets it up over
    vLLM's TP group: the peer buffer every rank maps, the launch counters and the health words."""
    import torch
    import torch.distributed as dist

    from aklr.plugins.vllm.models.kimi_k3.latent_moe import runtime

    device = torch.device("cuda", torch.cuda.current_device())
    if runtime.setup(dist.group.WORLD, dist.get_rank(), dist.get_world_size(), device) is None:
        raise RuntimeError("latent-MoE tail: the group's peer memory could not be set up")
