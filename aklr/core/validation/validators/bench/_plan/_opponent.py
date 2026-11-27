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

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.validators.bench._plan._threshold import Threshold
from aklr.core.validation.validators.report._finding import FindingCode, FindingRaised
from aklr.core.validation.validators.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.validators.bench._measure._op_bench import OpBench
    from aklr.core.validation.validators.bench._measure._result import Comparison

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
        candidate = bench.candidate
        baseline = ContractRegistry.default().get(candidate.schema.contract_id).baseline
        import torch

        own = {n: v.clone() if torch.is_tensor(v) else v for n, v in inputs.items()}
        call = candidate.call(own, key.axes)
        try:
            baseline.baseline(**call)
        except NotImplementedError as error:
            detail = f"the {self.baseline} Baseline declines this workload"
            raise FindingRaised(FindingCode.BASELINE_UNBOUND, detail) from error
        except Exception as error:  # noqa: BLE001 -- the Baseline's, never the Candidate's
            detail = f"the Baseline {self.baseline} raised {error!r}"
            raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
        return lambda: baseline.baseline(**call)

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        return None if comparison is None else comparison.speedup

    def kernel_microseconds(self, comparison: Comparison | None) -> float | None:
        if comparison is None:
            return None
        return float(f"{statistics.median(comparison.candidate_ms) * 1000:.3g}")


@dataclass(frozen=True)
class IncumbentOpponent(Opponent):
    label: ClassVar[str] = "incumbent"
    baseline: str
    path: str
    schema: Schema
    stored_speedup: float | None

    @property
    def name(self) -> str:
        return self.path

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return bench.incumbent(self.path, self.schema).bind(inputs, key.axes)

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        if comparison is None or self.stored_speedup is None:
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
