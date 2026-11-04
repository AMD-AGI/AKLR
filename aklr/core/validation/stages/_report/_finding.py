###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from enum import StrEnum
from typing import Final

from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages._report._markdown import (
    axes_cell,
    detail_text,
    one_line,
    schema_short_name,
)

__all__ = ["Finding", "FindingCode"]


class FindingCode(StrEnum):
    """Why a stage did not pass: the candidate's fault, or the platform's."""

    SCHEMA_INVALID = "schema_invalid"
    SOURCE_REJECTED = "source_rejected"
    PERF_EDITED = "perf_edited"
    BUILD_FAILED = "build_failed"
    CANDIDATE_CRASH = "candidate_crash"
    CANDIDATE_TIMEOUT = "candidate_timeout"
    CAPTURE_UNSAFE = "capture_unsafe"
    INPUT_MUTATION = "input_mutation"
    NUMERIC_MISMATCH = "numeric_mismatch"
    POST_TIMING_MISMATCH = "post_timing_mismatch"
    NOT_FASTER = "not_faster"
    REGRESSED = "regressed"
    NO_QUALIFYING_WORKLOAD = "no_qualifying_workload"
    BASELINE_FAILED = "baseline_failed"
    PLATFORM_FAILURE = "platform_failure"
    NO_MATCHING_DEVICE = "no_matching_device"

    @property
    def is_platform_fault(self) -> bool:
        return self in _PLATFORM_FAULT_CODES


_PLATFORM_FAULT_CODES: Final[frozenset[FindingCode]] = frozenset(
    {
        FindingCode.PLATFORM_FAILURE,
        FindingCode.NO_MATCHING_DEVICE,
    }
)


@frozen_dataclass
class Finding:
    code: FindingCode
    schema_path: str | None
    axes: dict[str, int] | None
    detail: str

    @classmethod
    def unlocated(cls, code: FindingCode, detail: str) -> Finding:
        return cls(code=code, schema_path=None, axes=None, detail=detail)

    def markdown_item(self) -> str:
        where = schema_short_name(self.schema_path or "-")
        return f"`{self.code}` {where} / {axes_cell(self.axes)}: {detail_text(self.detail)}"

    def plain_item(self) -> str:
        where = f" {self.schema_path}" if self.schema_path else ""
        return f"{self.code}{where}: {one_line(self.detail)}"
