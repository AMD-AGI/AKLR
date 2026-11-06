# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["mega_moe_group_init"]


def mega_moe_group_init() -> None:
    """mori's symmetric heap over the group, as aiter's MoriAll2AllManager sets it up:
    the kernels and MegaMoEV2 allocate their buffers there."""
    import os

    # the bench holds MegaMoEV2's buffers and the kernel's at once: more than mori's 4 GiB
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    import mori
    import torch
    import torch.distributed as dist

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
