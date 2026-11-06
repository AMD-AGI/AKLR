# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e report: each scenario's accuracy with AKLR off and on."""

from __future__ import annotations

import re
from typing import Literal

from packaging.version import InvalidVersion, Version

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._e2e._cookbook import MatchValue
from aklr.core.validation.stages._report._finding import Finding, FindingCode
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

__all__ = ["E2eRecord", "E2eReport", "ModelEvaluationResult"]


class ModelEvaluationResult(FrozenModel):
    """A model evaluation with AKLR enabled or disabled: scores, answers and serving failures."""

    argvs: tuple[str, ...] = ()
    gsm8k_score: float | None = None
    n_failed_requests: int = 0  # GSM8K's, each scored wrong
    # the on side's only
    prompt_answer: str | None = None
    answer_matches: bool | None = None
    # at least these: a killed server's last counts may be unsaved
    n_hit_calls: int | None = None
    n_fallback_calls: int | None = None
    failure: str = ""


class E2eRecord(FrozenModel):
    """One scenario, AKLR off against on, and what decides it."""

    scenario: str
    cookbook: str  # <vendor>/<model>
    cookbook_sha: str
    matches: dict[str, dict[str, MatchValue]]  # the cell of each role
    cookbook_images: tuple[str, ...] = ()
    installed_sglang: str
    notes: str = ""  # the cells' cautions
    # a fault that is no finding: off failing where the cookbook names another sglang
    warning: str = ""
    off: ModelEvaluationResult | None = None
    on: ModelEvaluationResult | None = None

    @property
    def is_other_sglang(self) -> bool:
        """Whether an image the cookbook names has another sglang release than the installed one."""
        try:
            installed = Version(self.installed_sglang).release
        except InvalidVersion:
            return False
        tags = (re.search(r":v(\d+(?:\.\d+)+)", image) for image in self.cookbook_images)
        return any(tag is not None and Version(tag[1]).release != installed for tag in tags)

    def off_fault(self, min_off_score: float) -> str:
        """Why off is no baseline (it failed, or scored below ``min_off_score``); empty when it is one."""
        off = self.off
        if off is None or (
            not off.failure and off.gsm8k_score is not None and off.gsm8k_score >= min_off_score
        ):
            return ""
        return (
            off.failure or f"GSM8K {off.gsm8k_score}, below {min_off_score}: the recipe or grading is broken"
        )

    def findings(self, *, max_drop: float, min_off_score: float) -> list[Finding]:
        """The verdict: off failing is the platform's (only a warning when the cookbook names
        another sglang), on crashing, scoring more than ``max_drop`` below off, or answering
        the prompt wrongly is AKLR's."""
        off, on = self.off, self.on
        if off is None:
            return []
        if fault := self.off_fault(min_off_score):
            # the warning (``warning``) says it instead
            if self.is_other_sglang:
                return []
            return [Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{self.scenario} off: {fault}")]
        if on is None:
            return []
        findings = []
        if on.failure:
            findings.append(Finding.unlocated(FindingCode.E2E_CRASH, f"{self.scenario} on: {on.failure}"))
        # rounded: a drop of exactly max_drop is not more than it
        if (difference := self.difference) is not None and round(difference, 9) < -max_drop:
            detail = (
                f"{self.scenario}: GSM8K on {on.gsm8k_score:.4f} is {-difference:.4f} below"
                f" off {off.gsm8k_score:.4f}, more than {max_drop}"
            )
            findings.append(Finding.unlocated(FindingCode.E2E_ACCURACY_DROP, detail))
        if on.answer_matches is False:
            detail = f"{self.scenario}: answered {on.prompt_answer!r}"
            findings.append(Finding.unlocated(FindingCode.E2E_WRONG_ANSWER, detail))
        return findings

    @property
    def difference(self) -> float | None:
        """On's GSM8K score less off's."""
        if self.off is None or self.on is None or self.off.gsm8k_score is None or self.on.gsm8k_score is None:
            return None
        return self.on.gsm8k_score - self.off.gsm8k_score

    def table_row(self) -> tuple[str, ...]:
        off, on = self.off or ModelEvaluationResult(), self.on or ModelEvaluationResult()
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
            detail_text(
                "; ".join(
                    filter(
                        None,
                        [
                            f"warning: {self.warning}" if self.warning else "",
                            self.notes,
                            *self.cookbook_images,
                            "AKLR served no call" if on.n_hit_calls == 0 else "",
                        ],
                    )
                )
            )
            or "-",  # fmt: skip
        )

    def console_line(self) -> str:
        off, on = self.off or ModelEvaluationResult(), self.on or ModelEvaluationResult()
        parts = [f"GSM8K off {off.gsm8k_score}", f"on {on.gsm8k_score}"]
        if failed := off.n_failed_requests + on.n_failed_requests:
            parts.append(
                f"{failed} GSM8K requests failed (off {off.n_failed_requests}, on {on.n_failed_requests})"
            )
        if self.warning:
            parts.append(f"warning: {one_line(self.warning)}")
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
        columns = (
            "scenario",
            "cookbook",
            "GSM8K off",
            "GSM8K on",
            "on - off",
            "answer",
            "hit calls",
            "fallback calls",
            "notes",
        )
        blocks: list[Block] = [Table(columns, tuple(r.table_row() for r in self.scenario_records))]
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
            cells = "; ".join(f"{role} {escape(match)}" for role, match in record.matches.items())
            blocks += [
                Heading(4, escape(record.scenario)),
                Paragraph(
                    f"cookbook {escape(record.cookbook)} at {record.cookbook_sha or '-'}, cells: {cells};"
                    f" installed sglang {escape(record.installed_sglang)}"
                ),
            ]
            blocks.append(BulletList(tuple(items)))
        return blocks
