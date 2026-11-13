# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    PositiveFloat,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from aklr.core.contract.semantics._frozen_model import FrozenModel


class WorkloadSpec(FrozenModel):
    """One item of a schema's ``workloads``: var axes, scalars and measurement."""

    axes: dict[str, int]  # every var axis of the schema, contract order; {}: none
    # every scalar input present in the schema, its order; {}: none. Smart union keeps bool/int/float types
    scalars: dict[str, bool | int | float] = {}  # noqa: RUF012
    # validation's correctness verdict on this workload; None: not measured
    accuracy: Literal["passed", "failed"] | None = None
    # over aiter (validation's formula); None: not measured
    speedup: PositiveFloat | None = None
    # the kernel's time per call under validation's graph timer; None: not measured
    microseconds: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_measured(self) -> WorkloadSpec:
        if self.speedup is not None and self.accuracy is None:
            raise ValueError(f"speedup {self.speedup} has no accuracy")
        if self.microseconds is not None and self.accuracy is None:
            raise ValueError(f"microseconds {self.microseconds} has no accuracy")
        if self.accuracy == "passed" and self.speedup is None:
            raise ValueError("a passed accuracy has no speedup")
        return self

    def qualifies(self, min_speedup: float) -> bool:
        """Whether dispatch may serve this workload at ``AKLR_MIN_SPEEDUP``: never a
        failed one; any when ``min_speedup <= 0``, else one whose speedup reaches it."""
        return self.accuracy != "failed" and (min_speedup <= 0 or (self.speedup or 0) >= min_speedup)

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.scalars:
            data.pop("scalars", None)
        if self.accuracy is None:
            data.pop("accuracy", None)
        if self.speedup is None:
            data.pop("speedup", None)
        if self.microseconds is None:
            data.pop("microseconds", None)
        return data


def axes_text(axes: Mapping[str, int]) -> str:
    """``m=1,n=32``; also the key joining measurements to workloads, safe since axis
    names never contain ``,`` or ``=``: do not reformat."""
    return ",".join(f"{a}={v}" for a, v in axes.items())
