###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The input sampler: a workload's input sets drawn by its schema's distributions, each rank and trial from its
own seed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aklr.core.contract.op import OpContract
from aklr.core.validation.stages.bench._measure._contract_workload import ContractWorkload

__all__ = ["InputSampler"]


@dataclass(frozen=True)
class InputSampler:
    schema: OpContract
    axes: dict[str, int]  # the workload's

    def draw(self, trial: int, rank: int, device: Any) -> dict[str, Any]:
        """Rank's input set of trial, row-major on device, on the seed of that rank and trial, the same in every
        process. Rank 0's seed names no rank, so a single rank draws as rank 0 of a group."""
        return self.draw_group(trial, rank + 1, device)[rank]

    def draw_group(self, trial: int, ranks: int, device: Any) -> list[dict[str, Any]]:
        """Every rank's input set of trial, each drawn here."""
        workload = ContractWorkload.of(self.schema, self.axes)
        return [workload.draw(f"trial {trial}", device, rank) for rank in range(ranks)]
