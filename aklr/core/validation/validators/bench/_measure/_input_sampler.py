###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aklr.core.contract.op import OpContract
from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload

__all__ = ["InputSampler"]


@dataclass(frozen=True)
class InputSampler:
    schema: OpContract
    axes: dict[str, int]

    def draw(self, trial: int, rank: int, device: Any) -> dict[str, Any]:
        return self.draw_group(trial, rank + 1, device)[rank]

    def draw_group(self, trial: int, ranks: int, device: Any) -> list[dict[str, Any]]:
        workload = ContractWorkload.of(self.schema, self.axes)
        return [workload.draw(f"trial {trial}", device, rank) for rank in range(ranks)]
