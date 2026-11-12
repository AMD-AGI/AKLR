###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A stage finding: whose fault, where, and what it means for the workload."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages.report._markdown import detail_text, escape, one_line, schema_short_name
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["Fault", "Finding", "FindingCode", "FindingRaised", "Outcome"]


class Fault(StrEnum):
    """Whose fault a finding is: a candidate's or schema's fails the stage, the platform's aborts it."""

    CANDIDATE = "candidate"
    SCHEMA = "schema"
    PLATFORM = "platform"


class Outcome(StrEnum):
    """What a finding means for its workload."""

    ACCURACY_FAILED = "accuracy_failed"
    PERFORMANCE_FAILED = "performance_failed"
    NOT_MEASURED = "not_measured"


class FindingCode(StrEnum):
    """Why a stage did not pass, or why a workload was demoted; each code says whose fault and what."""

    fault: Fault
    outcome: Outcome

    def __new__(cls, value: str, fault: Fault, outcome: Outcome) -> FindingCode:
        member = str.__new__(cls, value)
        member._value_ = value
        member.fault = fault
        member.outcome = outcome
        return member

    SCHEMA_INVALID = "schema_invalid", Fault.CANDIDATE, Outcome.NOT_MEASURED
    SOURCE_REJECTED = "source_rejected", Fault.CANDIDATE, Outcome.NOT_MEASURED
    PERF_EDITED = "perf_edited", Fault.CANDIDATE, Outcome.NOT_MEASURED
    # the bench: build_failed, task_crashed and task_timeout stand for every workload of their batch
    BUILD_FAILED = "build_failed", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    TASK_CRASHED = "task_crashed", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    TASK_TIMEOUT = "task_timeout", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    CANDIDATE_RAISED = "candidate_raised", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    WRONG_OUTPUTS = "wrong_outputs", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    OUTPUTS_MISMATCH = "outputs_mismatch", Fault.CANDIDATE, Outcome.ACCURACY_FAILED
    REGRESSED = "regressed", Fault.CANDIDATE, Outcome.PERFORMANCE_FAILED
    NOT_FASTER = "not_faster", Fault.CANDIDATE, Outcome.PERFORMANCE_FAILED
    NO_ADMITTED_WORKLOAD = "no_admitted_workload", Fault.CANDIDATE, Outcome.NOT_MEASURED
    BASELINE_UNBOUND = "baseline_unbound", Fault.SCHEMA, Outcome.NOT_MEASURED
    PLATFORM_FAULT = "platform_fault", Fault.PLATFORM, Outcome.NOT_MEASURED

    @property
    def is_platform_fault(self) -> bool:
        return self.fault == Fault.PLATFORM

    @property
    def is_demotion(self) -> bool:
        """A workload's own failure: recorded on it, and it does not decide the stage."""
        return self.outcome != Outcome.NOT_MEASURED

    @contextmanager
    def failing_every_rank(self, context: str) -> Iterator[None]:
        """A step every rank of the task's process group leaves the same way. An error in it is a finding of this
        code, its detail led by context; a FindingRaised in it keeps its own. When any rank's step fails,
        every rank raises the lowest failing rank's finding, its detail prefixed with that rank; a single rank
        raises its own."""
        import torch.distributed as dist

        failure = None
        try:
            yield
        except FindingRaised as error:
            failure = error
        except Exception as error:  # noqa: BLE001 -- any error of the step is its finding
            failure = FindingRaised(self, f"{context}: {type(error).__name__}: {error}")
            failure.__cause__ = error
        failures: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(failures, None if failure is None else (failure.code, failure.detail))
        for rank, failed in enumerate(failures):
            if failed is None:
                continue
            if len(failures) == 1:
                raise failure  # type: ignore[misc] -- this rank's own
            failed_code, detail = failed
            raise FindingRaised(failed_code, f"rank {rank}: {detail}") from failure


class FindingRaised(Exception):
    """A step stopped by a finding: its code and detail, wherever the step is."""

    def __init__(self, code: FindingCode, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@frozen_dataclass
class Finding:
    code: FindingCode
    schema_path: str | None
    workload: WorkloadKey | None
    detail: str

    @classmethod
    def unlocated(cls, code: FindingCode, detail: str) -> Finding:
        return cls(code=code, schema_path=None, workload=None, detail=detail)

    def markdown_item(self) -> str:
        where = schema_short_name(self.schema_path or "-")
        workload = escape(str(self.workload) if self.workload else "") or "-"
        return f"`{self.code}` {where} / {workload}: {detail_text(self.detail)}"

    def plain_item(self) -> str:
        where = f" {self.schema_path}" if self.schema_path else ""
        return f"{self.code}{where}: {one_line(self.detail)}"
