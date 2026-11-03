# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Self,
    TypeVar,
    get_args,
)

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._report._finding import Finding
from aklr.core.validation.stages._report._markdown import (
    BulletList,
    Heading,
    Paragraph,
    detail_text,
    one_line,
)

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport
    from aklr.core.validation.stages._report._markdown import Block
    from aklr.core.validation.stages.validation_scope import ValidationScope

__all__ = [
    "STAGE_ORDER", "PipelineStage", "Progress", "StageName", "StageReport",
    "StageStatus",
]  # fmt: skip

StageName = Literal["audit", "bench"]
STAGE_ORDER: Final = get_args(StageName)
StageStatus = Literal["passed", "failed", "skipped", "aborted"]
_MAX_CONSOLE_FINDINGS: Final = 10


class StageReport(FrozenModel):
    """What one stage found and measured; subclasses add their records."""

    stage_name: StageName
    status: StageStatus
    detail: str = ""
    findings: tuple[Finding, ...] = ()
    elapsed_seconds: float = 0.0

    @classmethod
    def from_findings(cls, findings: Sequence[Finding], **records: Any) -> Self:
        """Any candidate fault fails the stage; else any platform fault aborts it."""
        if any(not f.code.is_platform_fault for f in findings):
            status = "failed"
        elif findings:
            status = "aborted"
        else:
            status = "passed"
        return cls(status=status, findings=tuple(findings), **records)

    @classmethod
    def skipped(cls, detail: str) -> Self:
        """The report of a stage not run, and why."""
        return cls(status="skipped", detail=detail)

    @classmethod
    def aborted(cls, detail: str) -> Self:
        """The report of a stage stopped by aklr's own defect."""
        return cls(status="aborted", detail=detail)

    @property
    def is_passed(self) -> bool:
        return self.status == "passed"

    def table_row(self) -> tuple[str, ...]:
        if self.status == "skipped":
            return (self.stage_name, f"skipped: {detail_text(self.detail)}", "-", "-")
        return (
            self.stage_name,
            self.status,
            str(len(self.findings)),
            f"{self.elapsed_seconds:.1f}",
        )

    def console_lines(self) -> list[str]:
        """The finish text for the console: status line, then the first findings."""
        if self.status == "skipped":
            return [f"skipped: {one_line(self.detail)}"]
        n = len(self.findings)
        detail = f" ({one_line(self.detail)})" if self.detail else ""
        elapsed = timedelta(seconds=round(self.elapsed_seconds))
        lines = [f"{self.status}{detail}, findings: {n}, {elapsed}"]
        lines += ["  " + f.plain_item() for f in self.findings[:_MAX_CONSOLE_FINDINGS]]
        if n > _MAX_CONSOLE_FINDINGS:
            lines.append(f"  ... and {n - _MAX_CONSOLE_FINDINGS} more, see report.md")
        return lines

    def markdown_blocks(self) -> list[Block]:
        if self.status == "skipped":
            return []
        blocks: list[Block] = [Heading(3, self.stage_name)]
        if self.status == "aborted":
            blocks.append(Paragraph(detail_text(self.detail)))
        blocks += self.record_blocks()
        blocks.append(BulletList(tuple(f.markdown_item() for f in self.findings)))
        return blocks

    def record_blocks(self) -> list[Block]:
        return []


@dataclass(frozen=True)
class Progress:
    """Line-atomic stderr progress under one tag: a stage name, or ``validate``."""

    tag: str = "validate"
    # one stderr, one lock for every tag
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def write(self, text: str) -> None:
        """One ``[HH:MM:SS] <tag>: <text>`` line; a dead stderr is ignored."""
        line = f"[{time.strftime('%H:%M:%S')}] {self.tag}: {text}\n"
        with self._lock, contextlib.suppress(OSError, ValueError, AttributeError):
            sys.stderr.write(line)
            sys.stderr.flush()

    def finish(self, report: StageReport) -> None:
        """Write the report's console lines; a failure to build them is logged."""
        try:
            lines = report.console_lines()
        except Exception:  # progress text never costs a report
            logging.getLogger(__name__).exception("progress text failed")
            return
        for line in lines:
            self.write(line)


ReportT = TypeVar("ReportT", bound=StageReport)


class PipelineStage(ABC, Generic[ReportT]):
    """One stage of the pipeline, answering a ``report_type``."""

    name: ClassVar[StageName]
    report_type: ClassVar[type[StageReport]]

    @abstractmethod
    def run(self, scope: ValidationScope, earlier: PipelineReport) -> ReportT:
        """The stage's own work on ``scope``; ``earlier`` is the run so far."""

    @property
    def progress(self) -> Progress:
        return Progress(self.name)

    def execute(self, scope: ValidationScope, earlier: PipelineReport) -> ReportT:
        """Run the stage timed; its own exception aborts it as aklr's defect."""
        start = time.monotonic()
        try:
            report = self.run(scope, earlier)
        except Exception as e:  # noqa: BLE001 -- aklr's defect: aborted, reported
            traceback.print_exc()
            report = self.report_type.aborted(f"internal: {type(e).__name__}: {e}")
        elapsed_seconds = time.monotonic() - start
        return report.model_copy(update={"elapsed_seconds": elapsed_seconds})
