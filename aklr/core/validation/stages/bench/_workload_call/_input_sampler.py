###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""An input sampler: how a workload's input sets are made, one per rank and trial."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aklr.core.schema import Schema
    from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["InputSampler"]


class InputSampler(ABC):
    """Planned data, carried into the task: every member pickles."""

    @classmethod
    def for_workload(cls, schema: Schema, key: WorkloadKey) -> InputSampler:
        """The sampler of schema's workload of key: the one rule choosing how a workload's input sets are made."""
        from aklr.core.validation.stages.bench._workload_call._init_sampler import InitSampler

        return InitSampler(schema, key.axes)

    @abstractmethod
    def draw(self, trial: int, rank: int, device: Any) -> dict[str, Any]:
        """Rank's input set of trial, on device."""

    def draw_group(self, trial: int, ranks: int, device: Any) -> list[dict[str, Any]]:
        """Every rank's input set of trial, each drawn here."""
        return [self.draw(trial, rank, device) for rank in range(ranks)]
