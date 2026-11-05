# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e report: each scenario's accuracy with AKLR off and on."""

from __future__ import annotations

from typing import Final, Literal

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._report._markdown import (
    Block,
    BulletList,
    Heading,
    Paragraph,
    Table,
    detail_text,
    escape,
    one_line,
)
from aklr.core.validation.stages.stage import StageReport

__all__ = ["E2eRecord", "E2eReport", "SideResult"]

_E2E_COLUMNS: Final = ("scenario", "cookbook", "GSM8K off", "GSM8K on", "on - off", "answer",
    "hit calls", "fallback calls", "notes")  # fmt: skip


class SideResult(FrozenModel):
    """One side (AKLR off or on) of a scenario: what it scored and how it was served."""

    argvs: tuple[str, ...] = ()
    gsm8k_score: float | None = None
    # the on side's only
    prompt_answer: str | None = None
    answer_matches: bool | None = None
    # at least these: a killed server's last counts may be unsaved
    n_hit_calls: int | None = None
    n_fallback_calls: int | None = None
    failure: str = ""


class E2eRecord(FrozenModel):
    """One scenario, AKLR off against on: facts only."""

    scenario: str
    cookbook_sha: str
    notes: str = ""  # the cells' cautions, the images the cookbook names
    off: SideResult | None = None
    on: SideResult | None = None

    @property
    def difference(self) -> float | None:
        """On's GSM8K score less off's."""
        if self.off is None or self.on is None or self.off.gsm8k_score is None or self.on.gsm8k_score is None:
            return None
        return self.on.gsm8k_score - self.off.gsm8k_score

    def table_row(self) -> tuple[str, ...]:
        off, on = self.off or SideResult(), self.on or SideResult()
        difference = self.difference
        return (
            escape(self.scenario),
            self.cookbook_sha[:12] or "-",
            "-" if off.gsm8k_score is None else f"{off.gsm8k_score:.3f}",
            "-" if on.gsm8k_score is None else f"{on.gsm8k_score:.3f}",
            "-" if difference is None else f"{difference:+.3f}",
            {True: "matched", False: "differs", None: "-"}[on.answer_matches],
            "-" if on.n_hit_calls is None else str(on.n_hit_calls),
            "-" if on.n_fallback_calls is None else str(on.n_fallback_calls),
            detail_text(self.notes) or "-",
        )

    def console_line(self) -> str:
        off, on = self.off or SideResult(), self.on or SideResult()
        parts = [f"GSM8K off {off.gsm8k_score}", f"on {on.gsm8k_score}"]
        if on.answer_matches is not None:
            parts.append(
                f"answer {'matched' if on.answer_matches else 'differs'}: {one_line(on.prompt_answer or '')!r}"
            )
        if on.n_hit_calls is not None:
            parts.append(f"{on.n_hit_calls} hit calls, {on.n_fallback_calls} fallback calls")
        return ", ".join(parts)


class E2eReport(StageReport):
    stage_name: Literal["e2e"] = "e2e"
    prompt: str = ""
    expected_answer: str = ""
    scenario_records: tuple[E2eRecord, ...] = ()

    def record_blocks(self) -> list[Block]:
        blocks: list[Block] = [Table(_E2E_COLUMNS, tuple(r.table_row() for r in self.scenario_records))]
        for record in self.scenario_records:
            items: list[str] = []
            for side_name, side in (("off", record.off), ("on", record.on)):
                if side is None:
                    continue
                if side.prompt_answer is not None:
                    items.append(
                        f"{side_name} answer `{escape(one_line(side.prompt_answer))}`,"
                        f" expected `{escape(self.expected_answer)}` to {escape(self.prompt)}"
                    )
                if side.failure:
                    items.append(f"{side_name} failed: {detail_text(side.failure)}")
                items += [f"{side_name}: `{escape(argv)}`" for argv in side.argvs]
            blocks += [Heading(4, escape(record.scenario)), Paragraph(f"cookbook {record.cookbook_sha}")]
            blocks.append(BulletList(tuple(items)))
        return blocks
