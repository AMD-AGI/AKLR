###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""How the bench runs mega_moe: on its expert-parallel group, over mori's symmetric heap."""

from __future__ import annotations

from aklr.core.validation.stages.bench._execution_policy import ExecutionPolicy

__all__ = ["MegaMoeExecutionPolicy"]


class MegaMoeExecutionPolicy(ExecutionPolicy, name="mega_moe"):
    @property
    def ranks_per_task(self) -> int:
        return self.schema.const_axes["ep_size"]

    def setup(self) -> None:
        """mori's symmetric heap over the task's process group, where the kernels and MegaMoEV2 allocate their buffers."""
        import os

        # the bench holds MegaMoEV2's buffers and the kernel's at once: more than mori's 4 GiB
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
        import mori
        import torch
        import torch.distributed as dist

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")
