###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The input sampler: a workload's input sets drawn by its schema's init, each rank and trial from its own seed."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from aklr.core.schema import Schema
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["InputSampler"]


@dataclass(frozen=True)
class InputSampler:
    schema: Schema
    axes: dict[str, int]  # the workload's

    def draw(self, trial: int, rank: int, device: Any) -> dict[str, Any]:
        """Rank's input set of trial, on device: the standard's init's on the seed of that rank and trial, the
        same in every process, given the axes it declares, the schema's Const axes among them. Rank 0's seed
        names no rank, so a single rank draws as rank 0 of a group."""
        init = None if self.schema.standard is None else self.schema.standard.init
        if init is None:
            raise ValueError(f"the schema {self.schema.name} has no init: no template with one names it")
        declared = inspect.signature(init).parameters
        axes = {a: v for a, v in {**self.schema.const_axes, **self.axes}.items() if a in declared}
        salt = f"trial {trial}" + (f" rank {rank}" if rank else "")
        return init(**axes, device=device, seed=WorkloadKey.seed(self.schema.name, self.axes, salt))

    def draw_group(self, trial: int, ranks: int, device: Any) -> list[dict[str, Any]]:
        """Every rank's input set of trial, each drawn here."""
        return [self.draw(trial, rank, device) for rank in range(ranks)]
