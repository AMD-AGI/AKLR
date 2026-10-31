# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Specification for workloads, which defines the input tensors for a kernel."""

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
    """A workload's measured speedup over one baseline at one commit or tag."""

    commit: Annotated[str, Field(pattern=r"^\S+$")]
    """The baseline's git commit or tag the speedup was measured against."""
    value: PositiveFloat
    """Speedup over the baseline (validation's formula)."""


@frozen_dataclass
class Evaluation:
    hardware: str
    accuracy: Literal["passed", "failed"]
    """Validation's correctness verdict on this workload."""
    speedup: dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], Speedup] = Field(default_factory=dict)
    """Baseline name (``aiter``, ``aklr``, ...) to its speedup; {}: none timed."""

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
    """One item of a schema's ``workloads``: var axes, inputs and measurement.

    Defines a specific instance of a computational workload with concrete
    values for all variable axes and specifications for all input data.
    This represents an executable configuration that can be benchmarked.
    """

    axes: dict[str, NonNegativeInt]
    """Every var axis of the schema to its concrete value, schema order; {}: none."""
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
        """Whether dispatch may serve this workload on ``target`` in place of ``baseline``
        at ``AKLR_MIN_SPEEDUP``: never a failed one; any when ``min_speedup <= 0``; else a
        passed one whose speedup over ``baseline`` reaches it, or that was not measured
        against ``baseline``."""
        evaluation = self.evaluation(target)
        if evaluation is None:
            return min_speedup <= 0
        if evaluation.accuracy == "failed":
            return False
        measured = evaluation.speedup.get(baseline)
        return min_speedup <= 0 or measured is None or measured.value >= min_speedup
