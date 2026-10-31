# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Final, Literal

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._report._markdown import Block, Table, escape
from aklr.core.validation.stages.stage import StageReport

__all__ = ["E2eRecord", "E2eReport"]

_E2E_COLUMNS: Final = ("scenario", "runs per side", "throughput ratio lower bound",
    "latency ratio lower bound", "hit calls", "fallback calls")  # fmt: skip


class E2eRecord(FrozenModel):
    """One scenario, aklr on against off: facts only."""

    scenario: str
    n_runs_per_side: int
    throughput_ratio_lower_bound: float
    # off/on: the decode step, or TTFT; higher is better
    latency_ratio_lower_bound: float
    n_hit_calls: int
    n_fallback_calls: int

    def table_row(self) -> tuple[str, ...]:
        return (
            escape(self.scenario),
            str(self.n_runs_per_side),
            f"{self.throughput_ratio_lower_bound:.4f}",
            f"{self.latency_ratio_lower_bound:.4f}",
            str(self.n_hit_calls),
            str(self.n_fallback_calls),
        )

    def console_line(self, min_ratio: float) -> str:
        return (
            f"throughput ratio lower bound {self.throughput_ratio_lower_bound:.4f}, gate >= {min_ratio:.4f},"
            f" latency ratio lower bound {self.latency_ratio_lower_bound:.4f},"
            f" {self.n_hit_calls} hit calls, {self.n_fallback_calls} fallback calls"
        )


class E2eReport(StageReport):
    stage_name: Literal["e2e"] = "e2e"
    scenario_records: tuple[E2eRecord, ...] = ()

    def record_blocks(self) -> list[Block]:
        rows = tuple(record.table_row() for record in self.scenario_records)
        return [Table(_E2E_COLUMNS, rows)]
