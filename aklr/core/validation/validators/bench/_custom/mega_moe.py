###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from aklr.core.contract.op import OpContract as Schema
from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload
from aklr.core.validation.validators.bench._measure._op_bench import OpBench

__all__ = ["MegaMoeBench"]


class MegaMoeBench(OpBench, name="mega_moe"):
    @classmethod
    def ranks(cls, schema: Schema) -> int:
        return ContractWorkload.of(schema, schema.workloads[0].axes).ranks

    @classmethod
    def isolate(cls, schema: Schema) -> bool:
        return True
