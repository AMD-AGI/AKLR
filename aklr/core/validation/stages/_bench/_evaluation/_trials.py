###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

from aklr.core.schema import Schema, Tensor
from aklr.core.schema._workload import Workload
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages._measure import (
    Threshold,
    Versus,
    derive_seed,
    sorted_key_json,
)
from aklr.core.validation.stages._report._finding import FindingCode
from aklr.core.validation.stages._report._markdown import axes_text

if TYPE_CHECKING:
    import torch

__all__ = ["N_POST_TIMING_SETS", "CheckName", "Step", "TrialRequest"]

N_POST_TIMING_SETS: Final = 4

CheckName = Literal[
    "first_capture",
    "numeric",
    "post_timing",
]


# codes at (index 0, index > 0): a first_capture replay on new inputs that mismatches is capture-unsafe
_MISMATCH_CODES: Final[Mapping[CheckName, tuple[FindingCode, FindingCode]]] = {
    "first_capture": (FindingCode.NUMERIC_MISMATCH, FindingCode.CAPTURE_UNSAFE),
    "numeric": (FindingCode.NUMERIC_MISMATCH, FindingCode.NUMERIC_MISMATCH),
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
    schema: Schema
    workload: Workload
    axes_text: str
    thresholds: dict[Versus, Threshold]
    incumbent: Schema | None
    # the served base kernel's path (this one too), exported as the baseline; None: the bound adapter is
    baseline: str | None

    @classmethod
    def of_workload(
        cls,
        schema_path: str,
        schema: Schema,
        workload: Workload,
        *,
        thresholds: dict[Versus, Threshold],
        incumbent: Schema | None,
        baseline: str | None,
    ) -> TrialRequest:
        return cls(
            schema_path=schema_path,
            schema=schema,
            workload=workload,
            axes_text=axes_text(workload.axes),
            thresholds=thresholds,
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
        modes = ["capture"] * (N_POST_TIMING_SETS - 1) + ["replay"]
        for i, mode in enumerate(modes):
            steps.append(Step("post_timing", i, f"post_timing/{i}", "timing", mode))
        return tuple(steps)

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = self.schema.get_output_shapes(self.workload.axes)
        return {o: tuple(s) for o, s in zip(self.schema.outputs, shapes, strict=True)}

    @property
    def output_bytes(self) -> int:
        return len(self.steps) * sum(
            math.prod(shape) * dtype_str_to_torch_dtype(self.schema.outputs[o].dtype).itemsize
            for o, shape in self.output_shapes.items()
        )

    @property
    def input_bytes(self) -> int:
        """One input set's bytes: every present tensor input at the workload."""
        values = {**self.schema.const_axes, **self.workload.axes}
        return sum(
            math.prod(values[d] for d in spec.dim_names) * dtype_str_to_torch_dtype(spec.dtype).itemsize
            for spec in self.schema.inputs.values()
            if isinstance(spec, Tensor) and spec.dtype is not None and set(spec.dim_names) <= values.keys()
        )

    @property
    def workload_hash(self) -> str:
        text = sorted_key_json([self.schema.name, self.workload.axes])
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def compose_call(self, step: Step | None, device: str | torch.device) -> dict[str, Any]:
        """The api call for ``step`` (None: timing), seeded so both tasks draw the same inputs."""
        import torch

        input_set = "timing" if step is None else step.input_set
        init = self.schema.source_to_function("init")
        if init is None:
            raise ValueError(f"{self.schema.name}: the schema has no init")
        axes = {**self.schema.const_axes, **self.workload.axes}
        params = inspect.signature(init).parameters
        call = init(
            **{n: v for n, v in axes.items() if n in params},
            device=str(device),
            seed=derive_seed(self.workload_hash, input_set),
        )
        for o, shape in self.output_shapes.items():
            dtype = dtype_str_to_torch_dtype(self.schema.outputs[o].dtype)
            call[o] = torch.empty(shape, dtype=dtype, device=device)
        return call
