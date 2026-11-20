###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""How the bench runs mega_moe: on its expert-parallel group, over mori's symmetric heap."""

from __future__ import annotations

from aklr.core.contract.op import OpContract as Schema
from aklr.core.validation.stages.bench._measure._contract_workload import ContractWorkload
from aklr.core.validation.stages.bench._measure._op_bench import OpBench

__all__ = ["MegaMoeBench"]


class MegaMoeBench(OpBench, name="mega_moe"):
    @classmethod
    def ranks(cls, schema: Schema) -> int:
        """Its expert-parallel group's, the same at every workload: the contract's ranks attr."""
        return ContractWorkload.of(schema, schema.workloads[0].axes).ranks

    @classmethod
    def isolate(cls, schema: Schema) -> bool:
        """Always, a single rank too: setup registers mori's process group, once per process."""
        return True

    @classmethod
    def setup(cls, schema: Schema) -> None:
        """mori's symmetric heap over the task's process group, where the kernels and MegaMoEV2 allocate their buffers."""
        import os

        # the bench holds MegaMoEV2's buffers and the kernel's at once: more than mori's 4 GiB
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
        import mori
        import torch
        import torch.distributed as dist

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")
