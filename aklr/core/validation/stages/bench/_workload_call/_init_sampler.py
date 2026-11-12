###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The init sampler: a workload's input sets drawn by its schema's init, each rank and trial from its own seed."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

from aklr.core.schema import Schema
from aklr.core.validation.stages.bench._workload_call._input_sampler import InputSampler

__all__ = ["InitSampler"]


@dataclass(frozen=True)
class InitSampler(InputSampler):
    schema: Schema
    axes: dict[str, int]  # the workload's

    def seed(self, trial: int, rank: int) -> int:
        """Rank's seed of trial, the same in every process. Rank 0's salt names no rank, so a single rank draws
        as rank 0 of a group."""
        salt = f"trial {trial}" + (f" rank {rank}" if rank else "")
        key = f"{self.schema.name}|{sorted(self.axes.items())}|{salt}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

    def draw(self, trial: int, rank: int, device: Any) -> dict[str, Any]:
        """The standard's init's input set on its seed, given the axes it declares, the schema's Const axes among
        them."""
        init = None if self.schema.standard is None else self.schema.standard.init
        if init is None:
            raise ValueError(f"the schema {self.schema.name} has no init: no template with one names it")
        declared = inspect.signature(init).parameters
        axes = {a: v for a, v in {**self.schema.const_axes, **self.axes}.items() if a in declared}
        return init(**axes, device=device, seed=self.seed(trial, rank))
