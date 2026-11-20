###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages.bench._plan._threshold import Threshold
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.stages.bench._measure._op_bench import OpBench
    from aklr.core.validation.stages.bench._measure._result import Comparison

__all__ = ["BaselineOpponent", "IncumbentOpponent", "Opponent", "Unopposed"]


@dataclass(frozen=True)
class Opponent(ABC):
    label: ClassVar[str]
    baseline: str | None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None: ...

    def admits(self, comparison: Comparison | None) -> bool:
        return comparison is not None and comparison.speedup >= Threshold.ENTRY.bar

    @abstractmethod
    def baseline_speedup(self, comparison: Comparison | None) -> float | None: ...

    def kernel_microseconds(self, comparison: Comparison | None) -> float | None:
        return None


@dataclass(frozen=True)
class BaselineOpponent(Opponent):
    label: ClassVar[str] = "baseline"
    baseline: str

    @property
    def name(self) -> str:
        return self.baseline

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        import torch

        schema = bench.candidate.schema
        all_axes = {**schema.const_axes, **key.axes}
        with_outputs = dict(inputs)
        for (name, spec), shape in zip(
            schema.outputs.items(), schema.get_output_shapes(all_axes), strict=True
        ):
            if isinstance(spec, Tensor) and shape is not None:
                dtype = dtype_str_to_torch_dtype(spec.dtype)
                with_outputs.setdefault(name, torch.empty(shape, dtype=dtype, device=bench.device))
        baseline = schema.standard.baseline
        try:
            run = baseline(**with_outputs)
        except Exception as error:  # noqa: BLE001
            detail = f"the Baseline {baseline.__name__} raised {error!r}"
            raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
        if run is None:
            raise FindingRaised(
                FindingCode.BASELINE_UNBOUND, f"the {self.baseline} Baseline declines this workload"
            )
        return run

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        return None if comparison is None else comparison.speedup

    def kernel_microseconds(self, comparison: Comparison | None) -> float | None:
        if comparison is None:
            return None
        return float(f"{statistics.median(comparison.candidate_ms) * 1000:.3g}")


@dataclass(frozen=True)
class IncumbentOpponent(Opponent):
    label: ClassVar[str] = "incumbent"
    path: str
    schema: Schema
    stored_speedup: float | None

    @property
    def name(self) -> str:
        return self.path

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return bench.incumbent(self.path, self.schema).bind(inputs, key.axes)

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        if comparison is None or self.stored_speedup is None or self.baseline is None:
            return None
        return max(round(comparison.speedup * self.stored_speedup, 3), 0.001)


@dataclass(frozen=True)
class Unopposed(Opponent):
    label: ClassVar[str] = "nothing"
    baseline: None = None

    @property
    def name(self) -> str:
        return ""

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return None

    def admits(self, comparison: Comparison | None) -> bool:
        return True

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        return None
