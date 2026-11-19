###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A schema's workload as the contract layer reads it: its Point, its input sets and its api call."""

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
    # the whole json the semantics read (the file holds the values at the Point)
    schema_json: dict[str, Any]
    workload_json: dict[str, Any]
    # each present tensor's distribution item, evaluated: schema.distributions_at
    distributions: dict[str, dict[str, Any]]

    @classmethod
    def of(cls, schema: OpContract, axes: Mapping[str, int]) -> ContractWorkload:
        """schema's workload at axes, under the registered contract of its id."""
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
        """Ranks that call the op together at its Point."""
        return self.contract.ranks_at(self.point)

    @property
    def input_bytes(self) -> int:
        """One input set's bytes: every present tensor input at the Point."""
        dtypes = self.point.dtypes
        return sum(
            math.prod(self.contract.storage_shape_of(name, self.point))
            * max(semantics.DTYPE_TRAITS[dtypes[name]].bits, 8)
            // 8
            for name in self.distributions
        )

    def has_error_bound(self, kind: str) -> bool:
        return any(
            o["error_bounds"] and o["error_bounds"][0][1]["kind"] == kind
            for o in semantics.mutable_tensor_declarations(self.schema_json).values()
        )

    def draw(self, input_set: str, device: Any, rank: int = 0) -> dict[str, Any]:
        """Rank's row-major inputs of input_set, drawn on device: the same in every process."""
        return semantics.row_major_inputs(
            self.schema_json,
            self.workload_json,
            self.distributions,
            input_set=input_set,
            device=str(device),
            rank=rank,
        )

    def assemble_call(self, row_major_args: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
        """The api's call of row_major_args on device: the inputs as stored, the Point's attrs and layout facts,
        and each required output's buffer, new."""
        import torch

        contract, point = self.contract, self.point
        call = {p.name: p.default for p in contract.api_params}
        call.update(semantics.stored_inputs(self.schema_json, row_major_args, device=device))
        variant = contract.layout_variants.by_name[point.layout_variant_name]
        for fact in variant.detect_facts:
            param_name = fact.location.param_name(fact.name)
            if param_name is not None:
                call[param_name] = fact.expected_value
        for name, attr in contract.attrs.items():
            if attr.param_name is not None:
                call[attr.param_name] = point.attrs[name]
        if "out_dtype" in call:
            call["out_dtype"] = semantics.torch_dtypes(point.dtypes["out"])[0]
        for o in contract.required_outputs:
            dtype = semantics.torch_dtypes(point.dtypes[o])[0]
            shape = contract.storage_shape_of(o, point)
            call[o] = torch.empty(shape, dtype=dtype, device=device)
        return call
