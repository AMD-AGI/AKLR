###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's opponent: what its Candidate is timed against, and the speedup over the Baseline that yields."""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from aklr.core.contract.op import OpContract as Schema
from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages.bench._plan._threshold import Threshold
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.stages.bench._measure._op_bench import OpBench
    from aklr.core.validation.stages.bench._measure._result import Comparison

__all__ = ["BaselineOpponent", "IncumbentOpponent", "Opponent", "Unopposed"]


@dataclass(frozen=True)
class Opponent(ABC):
    """Planned data, carried into the task: every member pickles."""

    label: ClassVar[str]  # what the report says the workload was timed against
    # the Baseline's name: the contract's baseline function, dotted; None where the op has no Baseline
    baseline: str | None

    @property
    @abstractmethod
    def name(self) -> str:
        """Which one of its kind it is: the Baseline's name, the Incumbent's base path."""

    @abstractmethod
    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        """The opponent bound to a trial's inputs, ready to time; None when nothing is timed. Raises FindingRaised
        when it cannot be bound."""

    def admits(self, comparison: Comparison | None) -> bool:
        """Whether an accurate workload counts toward its schema's Admission: its published speedup reaches Entry,
        compared as dispatch compares a stored one."""
        return comparison is not None and comparison.speedup >= Threshold.ENTRY.bar

    @abstractmethod
    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        """The Candidate's speedup over the Baseline an accurate workload writes back; None when there is none."""

    def kernel_microseconds(self, comparison: Comparison | None) -> float | None:
        """The Candidate's median time per call an accurate workload writes back, in µs to 3 significant
        digits; None where it was not timed against the Baseline."""
        return None


@dataclass(frozen=True)
class BaselineOpponent(Opponent):
    label: ClassVar[str] = "baseline"
    baseline: str

    @property
    def name(self) -> str:
        return self.baseline

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        """The contract's baseline on the api's call of the inputs, called once here: one that raises
        NotImplementedError declines the workload."""
        candidate = bench.candidate
        baseline = ContractRegistry.default().get(candidate.schema.contract_id).baseline()
        # a Baseline writes into the output buffers it is handed, as the backend's caller hands them
        call = candidate.call(inputs, key.axes)
        try:
            baseline(**call)
        except NotImplementedError as error:
            detail = f"the {self.baseline} Baseline declines this workload"
            raise FindingRaised(FindingCode.BASELINE_UNBOUND, detail) from error
        except Exception as error:  # noqa: BLE001 -- the Baseline's, never the Candidate's
            detail = f"the Baseline {baseline.__name__} raised {error!r}"
            raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
        return lambda: baseline(**call)

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        return None if comparison is None else comparison.speedup

    def kernel_microseconds(self, comparison: Comparison | None) -> float | None:
        if comparison is None:
            return None
        return float(f"{statistics.median(comparison.candidate_ms) * 1000:.3g}")


@dataclass(frozen=True)
class IncumbentOpponent(Opponent):
    label: ClassVar[str] = "incumbent"
    path: str  # its base path
    schema: Schema
    stored_speedup: float | None  # over the Baseline on this target

    @property
    def name(self) -> str:
        return self.path

    def bind(self, bench: OpBench, inputs: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return bench.incumbent(self.path, self.schema).bind(inputs, key.axes)

    def baseline_speedup(self, comparison: Comparison | None) -> float | None:
        """The speedup over the Incumbent times its stored speedup: over the Baseline without timing it."""
        if comparison is None or self.stored_speedup is None or self.baseline is None:
            return None
        return max(round(comparison.speedup * self.stored_speedup, 3), 0.001)


@dataclass(frozen=True)
class Unopposed(Opponent):
    """Neither a Baseline nor an Incumbent: accuracy alone."""

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
