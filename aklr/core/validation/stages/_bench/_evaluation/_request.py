# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

from aklr.core.contract import semantics
from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import PointSpec, ScalarSpec, WorkloadSpec, axes_text
from aklr.core.validation.stages._measure import Threshold, Versus
from aklr.core.validation.stages._report._finding import FindingCode

if TYPE_CHECKING:
    import torch

__all__ = ["N_POST_TIMING_SETS", "CheckName", "Step", "TrialRequest"]

N_POST_TIMING_SETS: Final = 4

CheckName = Literal[
    "first_capture",
    "numeric",
    "large_magnitude",
    "determinism",
    "post_timing",
]


# a mismatching step's code: (index 0, index > 0); determinism has one step
_MISMATCH_CODES: Final[Mapping[CheckName, tuple[FindingCode, FindingCode]]] = {
    "first_capture": (FindingCode.NUMERIC_MISMATCH, FindingCode.CAPTURE_UNSAFE),
    "numeric": (FindingCode.NUMERIC_MISMATCH, FindingCode.NUMERIC_MISMATCH),
    "large_magnitude": (FindingCode.NUMERIC_MISMATCH, FindingCode.NUMERIC_MISMATCH),
    "determinism": (FindingCode.NUMERIC_MISMATCH, FindingCode.NUMERIC_MISMATCH),
    "post_timing": (FindingCode.POST_TIMING_MISMATCH,) * 2,
}


class Step(NamedTuple):
    """One api call of a workload, in the step table both tasks share."""

    check: CheckName
    index: int
    input_set: str
    buffer_source: Literal["new", "reuse", "timing"]
    mode: Literal["capture", "replay"]

    @property
    def name(self) -> str:
        return f"{self.check}/{self.index}"

    @property
    def is_out_reused(self) -> bool:
        return self.buffer_source == "reuse" or self.mode == "replay"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.input_set})"

    @property
    def mismatch_code(self) -> FindingCode:
        return _MISMATCH_CODES[self.check][min(self.index, 1)]

    def output_key(self, output: str) -> str:
        return f"{self.name}/{output}"


@dataclass(frozen=True)
class TrialRequest:
    """Everything a task needs of one workload, made in the parent and pickled to the child."""

    schema_path: str
    schema: OpContract
    contract: OpContract
    workload: WorkloadSpec
    point: PointSpec
    axes_text: str
    schema_json: dict[str, Any]
    workload_json: dict[str, Any]
    # each present tensor's distribution item, evaluated: schema.distributions_at
    distributions: dict[str, dict[str, Any]]
    thresholds: dict[Versus, Threshold]
    # the least published speedup over the baseline dispatch serves; 0: whatever is measured
    min_speedup: float
    incumbent: OpContract | None
    # what users run today, exported as the baseline: the base kernel dispatch serves
    # (this very kernel too), else the defined baseline's dotted name; None: neither
    baseline: str | None

    @classmethod
    def of_workload(
        cls,
        schema_path: str,
        schema: OpContract,
        contract: OpContract,
        workload: WorkloadSpec,
        *,
        thresholds: dict[Versus, Threshold],
        incumbent: OpContract | None,
        baseline: str | None,
        min_speedup: float = 0,
    ) -> TrialRequest:
        """The request of ``schema``'s ``workload``."""
        point = schema.point_with_contract_id(workload.axes, contract.contract_id)
        contract.check_point(point)
        workload_axes_text = axes_text(workload.axes)
        # the whole json the semantics read (the file holds the values at the Point)
        schema_json = schema.model_dump(mode="json")
        workloads_json = {axes_text(w["axes"]): w for w in schema_json["workloads"]}
        return cls(
            schema_path=schema_path,
            schema=schema,
            contract=contract,
            workload=workload,
            point=point,
            axes_text=workload_axes_text,
            schema_json=schema_json,
            workload_json=workloads_json[workload_axes_text],
            distributions=schema.distributions_at(point),
            thresholds=thresholds,
            min_speedup=min_speedup,
            incumbent=incumbent,
            baseline=baseline,
        )

    @property
    def steps(self) -> tuple[Step, ...]:
        steps = [Step("first_capture", 0, "numeric/0", "new", "capture")]
        for i in (1, 2):
            steps.append(Step("first_capture", i, f"numeric/{i}", "reuse", "replay"))
        for i in range(3):
            steps.append(Step("numeric", i, f"numeric/{i}", "new", "capture"))
        if self.has_workload_set:
            steps.append(Step("numeric", 3, "workload", "new", "capture"))
        if self.has_error_bound("dot_product"):
            steps.append(Step("large_magnitude", 0, "large_magnitude", "new", "capture"))
        if self.schema_json["compare"]["determinism"] == "bitwise":
            steps.append(Step("determinism", 0, "numeric/0", "new", "capture"))
        modes = ["capture"] * (N_POST_TIMING_SETS - 1) + ["replay"]
        for i, mode in enumerate(modes):
            steps.append(Step("post_timing", i, f"post_timing/{i}", "timing", mode))
        return tuple(steps)

    @property
    def has_workload_set(self) -> bool:
        inputs_json = self.schema_json["inputs"]
        return any(
            value
            != ScalarSpec.model_validate(inputs_json[name]).normalize(
                semantics.default_scalar(inputs_json[name])
            )
            for name, value in self.workload.scalars.items()
        )

    @property
    def output_bytes(self) -> int:
        dtypes = self.point.dtypes
        return len(self.steps) * sum(
            math.prod(self.contract.storage_shape_of(o, self.point))
            * max(semantics.DTYPE_TRAITS[dtypes[o]].bits, 8)
            // 8
            for o in self.contract.mutable_tensors
        )

    def empty_outputs(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        """Every step's mutable tensors, uninitialized on ``device``, by ``Step.output_key``
        (``output_bytes`` in all): the trial task lends them to the candidate's, which
        fills them."""
        import torch

        tensors = {
            o: (
                self.contract.storage_shape_of(o, self.point),
                semantics.torch_dtypes(self.point.dtypes[o])[0],
            )
            for o in self.contract.mutable_tensors
        }
        return {
            step.output_key(o): torch.empty(shape, dtype=dtype, device=device)
            for step in self.steps
            for o, (shape, dtype) in tensors.items()
        }

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

    def compose_inputs(self, step: Step | None, device: str) -> dict[str, Any]:
        input_set = "timing" if step is None else step.input_set
        row_major_args = semantics.row_major_inputs(
            self.schema_json,
            self.workload_json,
            self.distributions,
            input_set=input_set,
            device=device,
        )
        if step is None:
            return row_major_args
        if step.check == "large_magnitude":
            return semantics.magnified_inputs(self.schema_json, self.distributions, row_major_args)
        return row_major_args

    def assemble_call(self, row_major_args: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
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
