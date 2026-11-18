###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A schema's verdict: each workload's judgement (its findings, verdict, vote for Admission and write-back),
joined into the schema's."""

from __future__ import annotations

from collections.abc import Sequence

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import Evaluation, Speedup
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.bench._measure._result import Comparison, WorkloadResult
from aklr.core.validation.stages.bench._plan._opponent import Unopposed
from aklr.core.validation.stages.bench._plan._plan import SchemaPlan, WorkloadPlan
from aklr.core.validation.stages.bench._plan._threshold import Threshold
from aklr.core.validation.stages.report._finding import Finding, FindingCode, Outcome
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["SchemaVerdict", "WorkloadJudgement", "WorkloadRecord"]


@frozen_dataclass
class WorkloadRecord:
    """A workload's row of the report."""

    schema_path: str
    workload: WorkloadKey
    verdict: str  # its first finding's code, else "passed"
    is_demoted: bool
    versus: str  # its opponent's label
    versus_name: str  # the Baseline's name, or the Incumbent's base path
    threshold: str | None
    comparison: Comparison | None
    kernel_hash: str


class WorkloadJudgement:
    def __init__(
        self, plan: WorkloadPlan, comparison: Comparison | None, findings: tuple[Finding, ...]
    ) -> None:
        """findings: its task's, then its threshold's."""
        self.plan = plan
        self.comparison = comparison
        self.findings = findings

    @classmethod
    def judge(
        cls, schema_path: str, plan: WorkloadPlan, result: WorkloadResult, profile: MeasurementProfile
    ) -> WorkloadJudgement:
        """result judged against plan. A result that disagrees with its plan (timed where nothing is, untimed
        where an opponent is, or one side's timings not the other's) is the platform's: left alone, an untimed
        workload would be written back accurate with no speedup, which dispatch serves."""
        findings = list(result.findings)
        threshold = plan.threshold
        opponent = plan.opponent
        comparison = result.comparison(profile)
        n_candidate, n_opponent = len(result.candidate_ms), len(result.opponent_ms)
        if not findings and (
            n_candidate != n_opponent or (comparison is None) != isinstance(opponent, Unopposed)
        ):
            detail = (
                f"its task timed the Candidate {n_candidate} times and its opponent {n_opponent} times,"
                f" planned against {opponent.label}"
            )
            findings.append(
                Finding(
                    code=FindingCode.PLATFORM_FAULT, schema_path=schema_path, workload=plan.key, detail=detail
                )
            )
            comparison = None
        elif comparison is not None and not threshold.passes(comparison.ratio):
            detail = (
                f"ratio {comparison.ratio:.3f} over the {opponent.label} {opponent.name}"
                f" does not clear {threshold}"
            )
            findings.append(
                Finding(
                    code=threshold.failing_code, schema_path=schema_path, workload=plan.key, detail=detail
                )
            )
        return cls(plan, comparison, tuple(findings))

    @property
    def outcomes(self) -> set[Outcome]:
        return {FindingCode(f.code).outcome for f in self.findings}

    @property
    def is_accurate(self) -> bool:
        return Outcome.ACCURACY_FAILED not in self.outcomes

    @property
    def is_measured(self) -> bool:
        """Its accuracy is known: no finding of the platform's or the schema's stopped it first."""
        return Outcome.NOT_MEASURED not in self.outcomes

    @property
    def verdict(self) -> str:
        return self.findings[0].code if self.findings else "passed"

    @property
    def is_demoted(self) -> bool:
        return bool(self.findings) and FindingCode(self.findings[0].code).is_demotion

    def is_admitting(self) -> bool:
        """Whether it counts toward its schema's Admission."""
        return self.is_measured and self.is_accurate and self.plan.opponent.admits(self.comparison)

    def evaluation(self, target: DeviceTarget, head_sha: str) -> Evaluation:
        """What it writes back; an inaccurate workload gets no speedup."""
        opponent = self.plan.opponent
        value = opponent.baseline_speedup(self.comparison) if self.is_accurate else None
        speedup = {} if value is None else {opponent.baseline: Speedup(commit=head_sha, value=value)}
        accuracy = "passed" if self.is_accurate else "failed"
        microseconds = opponent.kernel_microseconds(self.comparison) if self.is_accurate else None
        return Evaluation(hardware=str(target), accuracy=accuracy, speedup=speedup, microseconds=microseconds)

    def record(self, schema_path: str, kernel_hash: str) -> WorkloadRecord:
        return WorkloadRecord(
            schema_path=schema_path,
            workload=self.plan.key,
            verdict=self.verdict,
            is_demoted=self.is_demoted,
            versus=self.plan.opponent.label,
            versus_name=self.plan.opponent.name,
            # where nothing is timed, no threshold applied
            threshold=None if isinstance(self.plan.opponent, Unopposed) else self.plan.threshold,
            comparison=self.comparison,
            kernel_hash=kernel_hash,
        )


@frozen_dataclass
class SchemaVerdict:
    path: str
    findings: tuple[Finding, ...]  # those that decide the stage
    demotions: tuple[Finding, ...]
    records: tuple[WorkloadRecord, ...]
    evaluations: dict[str, Evaluation] | None  # by workload key text; None: the schema is not written back
    skipped: str | None = None  # its plan's

    @classmethod
    def judge(
        cls,
        plan: SchemaPlan,
        results: Sequence[WorkloadResult],
        profile: MeasurementProfile,
        head_sha: str,
        target: DeviceTarget,
    ) -> SchemaVerdict:
        by_key = {str(r.key): r for r in results}
        judgements = [
            WorkloadJudgement.judge(
                plan.path,
                workload,
                by_key.get(str(workload.key))
                or WorkloadResult.failed(
                    plan.path, workload.key, FindingCode.PLATFORM_FAULT, "its task returned no result"
                ),
                profile,
            )
            for workload in plan.workloads
        ]
        findings = [f for j in judgements for f in j.findings]
        # an untimed plan writes nothing back, so it demotes nothing: every finding decides
        demotions = [f for f in findings if plan.is_timed and FindingCode(f.code).is_demotion]
        deciding = [*plan.findings, *(f for f in findings if f not in demotions)]
        is_admitted = any(j.is_admitting() for j in judgements)
        if plan.is_timed and plan.skipped is None and not deciding and not is_admitted:
            detail = f"no workload passed accuracy and cleared Entry ({Threshold.ENTRY.bar})"
            deciding.append(
                Finding(
                    code=FindingCode.NO_ADMITTED_WORKLOAD, schema_path=plan.path, workload=None, detail=detail
                )
            )
        is_written = (
            plan.is_timed
            and plan.skipped is None
            and all(j.is_measured for j in judgements)
            and all(f.code == FindingCode.NO_ADMITTED_WORKLOAD for f in deciding)
        )
        kernel = plan.schema.kernel
        kernel_hash = "" if kernel is None else kernel.hash()
        return cls(
            path=plan.path,
            findings=tuple(deciding),
            demotions=tuple(demotions),
            records=tuple(j.record(plan.path, kernel_hash) for j in judgements),
            evaluations=(
                {str(j.plan.key): j.evaluation(target, head_sha) for j in judgements} if is_written else None
            ),
            skipped=plan.skipped,
        )
