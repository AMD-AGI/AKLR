###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.op import OpContract
from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload

if TYPE_CHECKING:
    from aklr.core.runtime.dispatch import KernelDispatcher

__all__ = ["BuiltKernel"]


class BuiltKernel:
    def __init__(self, schema: OpContract, dispatcher: KernelDispatcher, device: Any) -> None:
        self.schema = schema
        self.dispatcher = dispatcher
        self.device = device

    @classmethod
    def build(cls, schema: OpContract, device: Any, kernel_schema: OpContract | None = None) -> BuiltKernel:
        import torch

        from aklr.core.contract.registry import ContractRegistry
        from aklr.core.runtime.builder import BuilderRegistry
        from aklr.core.runtime.dispatch import DispatchTable, KernelDispatcher

        contract = ContractRegistry.default().get(schema.contract_id)
        runnable = BuilderRegistry.get_instance().build(kernel_schema or schema)
        unmeasured = {"speedup": None, "accuracy": None, "microseconds": None}
        workloads = [w.model_copy(update=unmeasured) for w in schema.workloads]
        dispatcher = KernelDispatcher(contract)
        dispatcher.set_table(
            DispatchTable.from_schemas(
                contract, torch.device(device), [(schema, runnable, workloads)], min_speedup=0
            )
        )
        return cls(schema, dispatcher, device)

    def call(self, inputs: Mapping[str, Any], axes: Mapping[str, int]) -> dict[str, Any]:
        return ContractWorkload.of(self.schema, axes).assemble_call(inputs, self.device)

    def bind(self, inputs: Mapping[str, Any], axes: Mapping[str, int]) -> Callable[[], dict[str, Any]]:
        from aklr.core.runtime.dispatch import FallbackReason

        contract = self.dispatcher.contract
        call = self.call(inputs, axes)
        allocated = dict.fromkeys(contract.allocated_outputs)

        def run() -> dict[str, Any]:
            sent = call | allocated
            result = self.dispatcher.dispatch(sent)
            if isinstance(result, FallbackReason):
                raise RuntimeError(f"dispatch fell back: {result.value}")  # noqa: TRY004 -- not a type error
            return contract.call_mutable_tensors(result, sent)

        return run
