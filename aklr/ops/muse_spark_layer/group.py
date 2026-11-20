# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["muse_spark_layer_group_init"]


def muse_spark_layer_group_init() -> None:
    """mori's symmetric heap over the group: the kernels' and the baseline's peer
    buffers (the partial rows and flags each rank pushes to the others) live there."""
    import mori
    import torch
    import torch.distributed as dist

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
