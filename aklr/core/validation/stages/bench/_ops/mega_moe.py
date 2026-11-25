###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from aklr.core.schema import Schema
from aklr.core.validation.stages.bench._measure._op_bench import OpBench

__all__ = ["MegaMoeBench"]


class MegaMoeBench(OpBench, name="mega_moe"):
    @classmethod
    def ranks(cls, schema: Schema) -> int:
        return schema.const_axes["ep_size"]

    @classmethod
    def setup(cls, schema: Schema) -> None:
        import os

        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
        import mori
        import torch
        import torch.distributed as dist

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")
