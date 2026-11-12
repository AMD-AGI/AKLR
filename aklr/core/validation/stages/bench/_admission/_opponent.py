###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's opponent: what its Candidate is timed against, the bar it must clear, and the speedup it yields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aklr.core.schema import Speedup, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages.bench._admission._threshold import Threshold
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.bench._measurement._workload_result import Comparison
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised
from aklr.core.validation.stages.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.stages.bench._task._batch_run import BatchRun

__all__ = ["BaselineOpponent", "IncumbentOpponent", "Opponent", "Unopposed"]


@dataclass(frozen=True)
class Opponent(ABC):
    """Planned data, carried into the task: every member pickles."""

    threshold: Threshold | None  # None: nothing is timed, accuracy alone decides

    @property
    @abstractmethod
    def label(self) -> str:
        """What the report says the workload was timed against."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Which one of its kind it is: the Baseline's backend, the Incumbent's base path."""

    @abstractmethod
    def bind(self, batch_run: BatchRun, call: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        """The opponent bound to a trial's call, ready to time; None when nothing is timed. Raises FindingRaised
        when it cannot be bound."""

    def admits(self, comparison: Comparison | None, profile: MeasurementProfile) -> bool:
        """Whether an accurate workload counts toward its schema's Admission: its published speedup reaches Entry,
        compared as dispatch compares a stored one."""
        return comparison is not None and comparison.speedup >= profile.entry

    @abstractmethod
    def speedups(self, comparison: Comparison | None, head_sha: str) -> dict[str, Speedup]:
        """The speedups over the Baseline an accurate workload writes back, by the Baseline's name."""


@dataclass(frozen=True)
class BaselineOpponent(Opponent):
    backend: str  # the name its speedups are kept under: the seam backend dispatch serves in place of

    @property
    def label(self) -> str:
        return "baseline"

    @property
    def name(self) -> str:
        return self.backend

    def bind(self, batch_run: BatchRun, call: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        """The schema standard's baseline on the call. It imports the backend's function as this process has it:
        the bench enables neither dispatch nor tracing, so that is the backend's own, never AKLR's stand-in."""
        import torch

        schema = batch_run.candidate.schema
        all_axes = {**schema.const_axes, **key.axes}
        # a Baseline writes into the output buffers it is handed, as the backend's caller hands them
        with_outputs = dict(call)
        for (name, spec), shape in zip(
            schema.outputs.items(), schema.get_output_shapes(all_axes), strict=True
        ):
            if isinstance(spec, Tensor) and shape is not None:
                dtype = dtype_str_to_torch_dtype(spec.dtype)
                with_outputs.setdefault(name, torch.empty(shape, dtype=dtype, device=batch_run.device))
        baseline = schema.standard.baseline
        try:
            run = baseline(**with_outputs)
        except Exception as error:  # noqa: BLE001 -- the Baseline's, never the Candidate's
            detail = f"the Baseline {baseline.__name__} raised {error!r}"
            raise FindingRaised(FindingCode.PLATFORM_FAULT, detail) from error
        if run is None:
            raise FindingRaised(
                FindingCode.BASELINE_UNBOUND, f"the {self.backend} Baseline declines this workload"
            )
        return run

    def speedups(self, comparison: Comparison | None, head_sha: str) -> dict[str, Speedup]:
        return (
            {} if comparison is None else {self.backend: Speedup(commit=head_sha, value=comparison.speedup)}
        )


@dataclass(frozen=True)
class IncumbentOpponent(Opponent):
    path: str  # its base path
    text: str  # its schema
    stored_speedup: float | None  # over the Baseline on this target
    baseline: str | None  # the Baseline's name its stored speedup is keyed by

    @property
    def label(self) -> str:
        return "incumbent"

    @property
    def name(self) -> str:
        return self.path

    def bind(self, batch_run: BatchRun, call: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return batch_run.incumbent(self.path, self.text).bind(call, key.axes)

    def speedups(self, comparison: Comparison | None, head_sha: str) -> dict[str, Speedup]:
        """The ratio over the Incumbent times its stored speedup: keyed by the Baseline without timing it."""
        if comparison is None or self.stored_speedup is None or self.baseline is None:
            return {}
        value = max(round(comparison.speedup * self.stored_speedup, 3), 0.001)
        return {self.baseline: Speedup(commit=head_sha, value=value)}


@dataclass(frozen=True)
class Unopposed(Opponent):
    """Neither a Baseline nor an Incumbent: accuracy alone."""

    threshold: Threshold | None = None

    @property
    def label(self) -> str:
        return "nothing"

    @property
    def name(self) -> str:
        return ""

    def bind(self, batch_run: BatchRun, call: dict[str, Any], key: WorkloadKey) -> Callable[[], Any] | None:
        return None

    def admits(self, comparison: Comparison | None, profile: MeasurementProfile) -> bool:
        return True

    def speedups(self, comparison: Comparison | None, head_sha: str) -> dict[str, Speedup]:
        return {}
