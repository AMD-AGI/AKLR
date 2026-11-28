###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aklr.core.contract import semantics
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec, WorkloadSpec, axes_text

if TYPE_CHECKING:
    import torch

__all__ = ["ContractWorkload"]


@dataclass(frozen=True)
class ContractWorkload:
    schema: OpContract
    contract: OpContract
    workload: WorkloadSpec
    point: PointSpec
    schema_json: dict[str, Any]
    workload_json: dict[str, Any]
    distributions: dict[str, dict[str, Any]]

    @classmethod
    def of(cls, schema: OpContract, axes: Mapping[str, int]) -> ContractWorkload:
        contract = ContractRegistry.default().get(schema.contract_id)
        workload = next(w for w in schema.workloads if w.axes == dict(axes))
        point = schema.point_with_contract_id(workload.axes, contract.contract_id)
        contract.check_point(point)
        schema_json = schema.model_dump(mode="json")
        key = axes_text(workload.axes)
        return cls(
            schema=schema,
            contract=contract,
            workload=workload,
            point=point,
            schema_json=schema_json,
            workload_json=next(w for w in schema_json["workloads"] if axes_text(w["axes"]) == key),
            distributions=schema.distributions_at(point),
        )

    @property
    def ranks(self) -> int:
        return self.contract.ranks_at(self.point)

    @property
    def input_bytes(self) -> int:
        dtypes = self.point.dtypes
        return sum(
            math.prod(self.contract.storage_shape_of(name, self.point))
            * max(semantics.DTYPE_TRAITS[dtypes[name]].bits, 8)
            // 8
            for name in self.distributions
        )

    def draw(self, input_set: str, device: Any, rank: int = 0) -> dict[str, Any]:
        return semantics.row_major_inputs(
            self.schema_json,
            self.workload_json,
            self.distributions,
            input_set=input_set,
            device=str(device),
            rank=rank,
        ) | self.workload_json.get("scalars", {})

    def assemble_call(self, row_major_args: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
        return self._call(semantics.stored_inputs(self.schema_json, row_major_args, device=device), device)

    def reference_call(self, row_major_args: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
        return self._call(row_major_args, device)

    def _call(self, inputs: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
        import torch

        contract, point = self.contract, self.point
        call = {p.name: p.default for p in contract.api_params}
        call.update(inputs)
        variant = contract.layout_variants.by_name[point.layout_variant_name]
        for fact in variant.detect_facts:
            param_name = fact.location.param_name(fact.name)
            if param_name is not None:
                call[param_name] = fact.expected_value
        for name, attr in contract.attrs.items():
            if attr.param_name is not None:
                call[attr.param_name] = point.attrs[name]
        for o, spec in contract.outputs.items():
            if spec.dtype.param_name is not None:
                call[spec.dtype.param_name] = semantics.torch_dtypes(point.dtypes[o])[0]
        for o in contract.required_outputs:
            dtype = semantics.torch_dtypes(point.dtypes[o])[0]
            shape = contract.storage_shape_of(o, point)
            call[o] = torch.empty(shape, dtype=dtype, device=device)
        return call
