###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Final

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.validation.stages._report._markdown import Block, Table, escape
from aklr.core.validation.stages.stage import StageReport

__all__ = ["E2eRecord", "E2eReport"]

_E2E_COLUMNS: Final = ("scenario", "runs per side", "throughput ratio lower bound",
    "latency ratio lower bound")  # fmt: skip


@frozen_dataclass
class E2eRecord:
    """One scenario's result, aklr on against off."""

    scenario: str
    n_runs_per_side: int
    throughput_ratio_lower_bound: float
    latency_ratio_lower_bound: float  # off/on of the decode step, or TTFT: higher is better

    def table_row(self) -> tuple[str, ...]:
        return (
            escape(self.scenario),
            str(self.n_runs_per_side),
            f"{self.throughput_ratio_lower_bound:.4f}",
            f"{self.latency_ratio_lower_bound:.4f}",
        )

    def console_line(self, min_ratio: float) -> str:
        return (
            f"throughput ratio lower bound {self.throughput_ratio_lower_bound:.4f}, gate >= {min_ratio:.4f},"
            f" latency ratio lower bound {self.latency_ratio_lower_bound:.4f}"
        )


@frozen_dataclass
class E2eReport(StageReport):
    scenario_records: tuple[E2eRecord, ...] = ()

    def record_blocks(self) -> list[Block]:
        rows = tuple(record.table_row() for record in self.scenario_records)
        return [Table(_E2E_COLUMNS, rows)]
