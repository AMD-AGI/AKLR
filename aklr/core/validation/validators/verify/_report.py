# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, get_args

from pydantic import Field

from aklr.core.common.frozen_model import frozen_dataclass
from aklr.core.validation.validators.report._markdown import (
    Paragraph,
    Table,
    detail_text,
    escape,
    schema_short_name,
)
from aklr.core.validation.validators.validator import ValidatorReport
from aklr.core.validation.validators.verify._verdict import Verdict, VerdictStatus
from aklr.core.validation.validators.workload_key import WorkloadKey

if TYPE_CHECKING:
    from aklr.core.validation.validators.report._markdown import Block

__all__ = ["PathRecord", "VerifyReport"]


@frozen_dataclass
class PathRecord:
    schema_path: str
    workload: WorkloadKey
    kernel: str | None
    verdicts: tuple[Verdict, ...]


@frozen_dataclass
class VerifyReport(ValidatorReport, name="verify"):
    path_records: tuple[PathRecord, ...] = ()
    rules: dict[str, int] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)

    def rule_label(self, rule: str) -> str:
        return escape(f"VR{self.rules[rule]} {rule}" if rule in self.rules else rule)

    def console_lines(self) -> list[str]:
        lines = super().console_lines()
        counts = Counter(v.status for r in self.path_records for v in r.verdicts)
        if counts:
            lines.insert(1, "  " + ", ".join(f"{s} {counts[s]}" for s in get_args(VerdictStatus)))
        return lines

    def record_blocks(self) -> list[Block]:
        verdicts = [(r, v) for r in self.path_records for v in r.verdicts]
        if not verdicts:
            return []
        statuses = get_args(VerdictStatus)
        by_rule = Counter((v.rule, v.status) for _, v in verdicts)
        blocks: list[Block] = [
            Paragraph("; ".join(f"{name}: {escape(tool)}" for name, tool in self.tools.items())),
            Table(
                ("rule", *statuses),
                tuple(
                    (self.rule_label(rule), *(str(by_rule[rule, status]) for status in statuses))
                    for rule in dict.fromkeys(v.rule for _, v in verdicts)
                ),
            ),
        ]
        unchecked = Counter((v.rule, v.reason) for _, v in verdicts if v.status == "unchecked")
        if unchecked:
            rows = tuple(
                (self.rule_label(rule), escape(reason), str(n))
                for (rule, reason), n in unchecked.most_common()
            )
            blocks += [
                Paragraph("unchecked, neither passed nor failed:"),
                Table(("rule", "why", "paths"), rows),
            ]
        open_rows = tuple(
            (
                schema_short_name(r.schema_path),
                escape(str(r.workload)),
                escape(r.kernel or "-"),
                self.rule_label(v.rule),
                v.status,
                detail_text(v.detail),
            )
            for r, v in verdicts
            if v.status == "rejected"
        )
        if open_rows:
            blocks.append(Table(("schema", "workload", "kernel", "rule", "status", "detail"), open_rows))
        return blocks
