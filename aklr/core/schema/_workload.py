###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.platform.device import DeviceTarget


@frozen_dataclass
class Speedup:
    """A workload's speedup over one baseline, measured at that baseline's commit or tag."""

    commit: Annotated[str, Field(pattern=r"^\S+$")]
    value: PositiveFloat


@frozen_dataclass
class Evaluation:
    hardware: str
    accuracy: Literal["passed", "failed"]
    speedup: dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], Speedup] = Field(default_factory=dict)

    @field_validator("hardware")
    @classmethod
    def _validate_hardware(cls, value: str) -> str:
        arch, _, hardware = value.partition("/")
        target = DeviceTarget.from_names(arch, hardware)
        if not target.is_complete:
            raise ValueError(f"hardware {value!r} is not '<arch>/<hardware>'")
        return str(target)


@frozen_dataclass
class Workload:
    axes: dict[str, NonNegativeInt]
    evaluations: list[Evaluation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_evaluations(self) -> Workload:
        hardware = [e.hardware for e in self.evaluations]
        if len(set(hardware)) != len(hardware):
            raise ValueError(f"evaluations repeat a hardware: {hardware}")
        return self

    def evaluation(self, target: DeviceTarget) -> Evaluation | None:
        return next((e for e in self.evaluations if e.hardware == str(target)), None)

    def qualifies(self, target: DeviceTarget, min_speedup: float, baseline: str) -> bool:
        """Whether dispatch may serve this workload on target in place of baseline at AKLR_MIN_SPEEDUP."""
        evaluation = self.evaluation(target)
        if evaluation is None:
            return min_speedup <= 0
        if evaluation.accuracy == "failed":
            return False
        measured = evaluation.speedup.get(baseline)
        return min_speedup <= 0 or measured is None or measured.value >= min_speedup
