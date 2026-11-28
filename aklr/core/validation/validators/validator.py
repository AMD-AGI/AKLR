###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Self

from pydantic import SerializerFunctionWrapHandler, model_serializer, model_validator
from pydantic_core import ArgsKwargs

from aklr.core.common import Registrable
from aklr.core.common.frozen_model import frozen_dataclass
from aklr.core.validation.validators.report._finding import Finding, FindingCode
from aklr.core.validation.validators.report._markdown import (
    BulletList,
    Heading,
    Paragraph,
    detail_text,
    one_line,
)

if TYPE_CHECKING:
    from aklr.core.validation.report import PlatformRecord
    from aklr.core.validation.validators.report._markdown import Block
    from aklr.core.validation.validators.validation_scope import ValidationScope

__all__ = ["Progress", "Status", "Validator", "ValidatorReport"]

Status = Literal["passed", "failed", "skipped", "aborted"]
_MAX_CONSOLE_FINDINGS: Final = 10


@frozen_dataclass
class ValidatorReport(Registrable):
    status: Status
    detail: str = ""
    findings: tuple[Finding, ...] = ()
    elapsed_secs: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _read_named(cls, data: Any) -> Any:
        fields = data.kwargs if isinstance(data, ArgsKwargs) else data
        if not isinstance(fields, dict) or "name" not in fields:
            return data
        fields = dict(fields)
        if (name := fields.pop("name")) != cls.name:
            raise ValueError(f"the {cls.name} report cannot be named {name!r}")
        return ArgsKwargs(data.args, fields) if isinstance(data, ArgsKwargs) else fields

    @model_serializer(mode="wrap")
    def _named(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {"name": self.name, **handler(self)}

    @classmethod
    def from_findings(cls, findings: Sequence[Finding], **records: Any) -> Self:
        if any(not FindingCode(f.code).is_platform_fault for f in findings):
            status = "failed"
        elif findings:
            status = "aborted"
        else:
            status = "passed"
        return cls(status=status, findings=tuple(findings), **records)

    @classmethod
    def skipped(cls, detail: str) -> Self:
        return cls(status="skipped", detail=detail)

    @classmethod
    def aborted(cls, detail: str) -> Self:
        return cls(status="aborted", detail=detail)

    @property
    def is_passed(self) -> bool:
        return self.status == "passed"

    def table_row(self) -> tuple[str, ...]:
        if self.status == "skipped":
            return (self.name, f"skipped: {detail_text(self.detail)}", "-", "-")
        return (
            self.name,
            self.status,
            str(len(self.findings)),
            f"{self.elapsed_secs:.1f}",
        )

    def console_lines(self) -> list[str]:
        if self.status == "skipped":
            return [f"skipped: {one_line(self.detail)}"]
        n = len(self.findings)
        detail = f" ({one_line(self.detail)})" if self.detail else ""
        elapsed = timedelta(seconds=round(self.elapsed_secs))
        lines = [f"{self.status}{detail}, findings: {n}, {elapsed}"]
        lines += ["  " + f.plain_item() for f in self.findings[:_MAX_CONSOLE_FINDINGS]]
        if n > _MAX_CONSOLE_FINDINGS:
            lines.append(f"  ... and {n - _MAX_CONSOLE_FINDINGS} more, see report.md")
        return lines

    def markdown_blocks(self) -> list[Block]:
        if self.status == "skipped":
            return []
        blocks: list[Block] = [Heading(3, self.name)]
        if self.status == "aborted":
            blocks.append(Paragraph(detail_text(self.detail)))
        blocks += self.record_blocks()
        blocks.append(BulletList(tuple(f.markdown_item() for f in self.findings)))
        return blocks

    def record_blocks(self) -> list[Block]:
        return []


@dataclass(frozen=True)
class Progress:
    tag: str = "validate"
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def write(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {self.tag}: {text}\n"
        with self._lock, contextlib.suppress(OSError, ValueError, AttributeError):
            sys.stderr.write(line)
            sys.stderr.flush()

    def finish(self, report: ValidatorReport) -> None:
        try:
            lines = report.console_lines()
        except Exception:
            logging.getLogger(__name__).exception("progress text failed")
            return
        for line in lines:
            self.write(line)


class Validator(Registrable, ABC):
    @abstractmethod
    def run(self, scope: ValidationScope, platform: PlatformRecord) -> ValidatorReport: ...

    @property
    def report_type(self) -> type[ValidatorReport]:
        return ValidatorReport.resolve(self.name)

    @property
    def progress(self) -> Progress:
        return Progress(self.name)

    def execute(self, scope: ValidationScope, platform: PlatformRecord) -> ValidatorReport:
        start = time.monotonic()
        try:
            report = self.run(scope, platform)
        except Exception as e:  # noqa: BLE001 -- aklr's defect: aborted, reported
            traceback.print_exc()
            report = self.report_type.aborted(f"internal: {type(e).__name__}: {e}")
        elapsed_secs = time.monotonic() - start
        return replace(report, elapsed_secs=elapsed_secs)
