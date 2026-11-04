###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import html
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, Self

from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.schema._workload import Evaluation, Speedup
from aklr.core.validation.stages._measure import (
    THRESHOLD_RATIOS,
    TimingRecord,
    ratio_lower_bound,
)
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages._report._markdown import (
    Block,
    BulletList,
    Paragraph,
    Table,
    axes_cell,
    axes_text,
    schema_short_name,
)
from aklr.core.validation.stages.stage import StageReport

if TYPE_CHECKING:
    from aklr.core.schema import Schema

__all__ = ["BenchReport", "WorkloadRecord"]

_WORKLOAD_COLUMNS: Final = (
    "schema", "axes", "verdict", "graph/baseline", "graph/incumbent",
)  # fmt: skip

# the accuracy a workload finding records, even when another workload admits the kernel
_MEASURED_AS: Final[Mapping[FindingCode, Literal["passed", "failed"]]] = {
    FindingCode.NUMERIC_MISMATCH: "failed",
    FindingCode.POST_TIMING_MISMATCH: "failed",
    FindingCode.INPUT_MUTATION: "failed",
    FindingCode.CAPTURE_UNSAFE: "failed",
    FindingCode.BUILD_FAILED: "failed",
    FindingCode.CANDIDATE_CRASH: "failed",
    FindingCode.CANDIDATE_TIMEOUT: "failed",
    # the gate's codes: correct but slow; dispatch's speedup threshold drops them
    FindingCode.NOT_FASTER: "passed",
    FindingCode.REGRESSED: "passed",
}


@frozen_dataclass
class WorkloadRecord:
    schema_path: str
    axes: dict[str, int]
    kernel_content_hash: str
    # what users run today: the served base kernel's path, else the bound adapter's entry point; None: neither
    baseline: str | None
    backend: str | None
    # False: no adapter and no other kernel to beat, so nothing was timed
    is_timed: bool
    is_compared: bool
    timings: tuple[TimingRecord, ...]
    # a batch finding (build, task crash or timeout) stands in for the workload's own findings
    batch_stopped: bool

    @property
    def beats_baseline(self) -> bool:
        if self.batch_stopped:
            return False
        if not self.is_timed:
            return True
        baselines = [t for t in self.timings if t.versus == "baseline"]
        return {t.mode for t in baselines} == {"graph"} and all(
            math.isfinite(t.ratio_lower_bound) and t.ratio_lower_bound > THRESHOLD_RATIOS["entry"]
            for t in baselines
        )

    @property
    def speedup(self) -> float | None:
        """Unclamped graph/baseline ratio lower bound, floored at 0.001; 1.0 if untimed (its own baseline)."""
        if not self.is_timed and not self.batch_stopped:
            return 1.0
        baselines = [t for t in self.timings if t.versus == "baseline"]
        if {t.mode for t in baselines} != {"graph"}:
            return None
        lower_bounds = (
            ratio_lower_bound(
                [math.log(v / c) for v, c in zip(t.versus_milliseconds, t.candidate_milliseconds)]
            )
            for t in baselines
        )
        return max(round(min(lower_bounds), 3), 0.001)

    def table_row(self, verdict: str) -> tuple[str, ...]:
        ratios = {f"{t.mode}/{t.versus}": f"{t.ratio_lower_bound:.4f}" for t in self.timings}
        return (
            schema_short_name(self.schema_path),
            axes_cell(self.axes),
            verdict,
            *(ratios.get(column, "-") for column in _WORKLOAD_COLUMNS[3:]),
        )


@frozen_dataclass
class BenchReport(StageReport):
    workload_records: tuple[WorkloadRecord, ...] = ()
    demotions: tuple[Finding, ...] = ()

    @classmethod
    def from_findings(cls, stage_name: str, findings: Sequence[Finding], **records: Any) -> Self:
        """Measured workload findings become demotions; a schema with no qualifying workload fails."""
        workloads = records.get("workload_records", ())
        demotable = {(r.schema_path, axes_text(r.axes)) for r in workloads}
        incorrect = {
            (f.schema_path, axes_text(f.axes))
            for f in findings
            if f.axes is not None and f.code not in (FindingCode.NOT_FASTER, FindingCode.REGRESSED)
        }
        qualified = {
            r.schema_path
            for r in workloads
            if r.is_compared and (r.schema_path, axes_text(r.axes)) not in incorrect and r.beats_baseline
        }
        deciding: list[Finding] = []
        demoted: list[Finding] = []
        for f in findings:
            is_demotion = (
                f.axes is not None
                and (f.schema_path, axes_text(f.axes)) in demotable
                and f.code in _MEASURED_AS
            )
            if is_demotion:
                demoted.append(f)
            else:
                deciding.append(f)
        for path in sorted({r.schema_path for r in workloads} - qualified):
            if any(f.schema_path in (None, path) for f in deciding):
                continue
            deciding.append(
                Finding(
                    code=FindingCode.NO_QUALIFYING_WORKLOAD,
                    schema_path=path,
                    axes=None,
                    detail="no workload both passes correctness and has a graph/"
                    f"baseline ratio lower bound > {THRESHOLD_RATIOS['entry']:g}"
                    " (correctness alone when no baseline exists)",
                )
            )
        return super().from_findings(stage_name, deciding, demotions=tuple(demoted), **records)

    def verdicts(self) -> list[tuple[WorkloadRecord, Finding | None]]:
        # reversed: the dict keeps the last, so a workload's first finding is its verdict
        first = {
            (f.schema_path, frozenset(f.axes.items())): f
            for f in reversed((*self.findings, *self.demotions))
            if f.axes is not None
        }
        return [(r, first.get((r.schema_path, frozenset(r.axes.items())))) for r in self.workload_records]

    def verdict_text(self, finding: Finding | None) -> str:
        if finding is None:
            return "passed"
        if finding in self.demotions:
            return f"{finding.code} (demoted)"
        return str(finding.code)

    def speedups(self) -> list[float]:
        return sorted(s for r in self.workload_records if (s := r.speedup) is not None)

    def accuracy(self) -> tuple[int, int]:
        """(failed, ok) workload counts; the rest are not measured."""
        failed = ok = 0
        for r, v in self.verdicts():
            is_failed = v is not None and not r.batch_stopped and _MEASURED_AS.get(v.code) == "failed"
            failed += is_failed
            ok += not is_failed and r.is_compared
        return failed, ok

    def batch_line(self) -> str:
        n = len(self.workload_records)
        parts = [f"{n} wl"]
        s = self.speedups()
        if s:
            lo, hi = f"{s[0]:.2f}", f"{s[-1]:.2f}"
            parts.append(f"speedup {lo}x" if lo == hi else f"speedup {lo}x-{hi}x")
        failed, ok = self.accuracy()
        if failed:
            parts.append(f"acc FAIL {failed}/{n}")
        elif not ok:
            parts.append("acc -")
        elif ok == n:
            parts.append("acc ok")
        else:
            parts.append(f"acc ok {ok}/{n}")
        # schema_short_name escapes markdown as &#N; entities; a console line wants them plain
        label = html.unescape(schema_short_name(self.workload_records[0].schema_path))
        return f"{self.status} {label}: {', '.join(parts)}"

    def outcome_lines(self) -> list[str]:
        if not self.workload_records:
            return []
        n = len(self.workload_records)
        s = self.speedups()
        failed, ok = self.accuracy()
        return [
            f"{n} workloads",
            f"  {'outcome':<32}{'workloads':>9}",
            *(
                f"  {t:<32}{c:>9}"
                for t, c in Counter(self.verdict_text(v) for _, v in self.verdicts()).most_common()
            ),
            (
                f"  speedup  best {s[-1]:.2f}x, median {statistics.median(s):.2f}x,"
                f" worst {s[0]:.2f}x  (workloads with a speedup value)"
                if s
                else "  speedup  -"
            ),
            f"  accuracy FAIL {failed}, ok {ok}, not measured {n - failed - ok}",
            "  per-schema table: report.md",
        ]

    def console_lines(self) -> list[str]:
        return [*self.outcome_lines(), *super().console_lines()]

    def record_blocks(self) -> list[Block]:
        rows = tuple(r.table_row(self.verdict_text(v)) for r, v in self.verdicts())
        blocks: list[Block] = [Table(_WORKLOAD_COLUMNS, rows)]
        if self.demotions:
            blocks += [
                Paragraph("demoted:"),
                BulletList(tuple(f.markdown_item() for f in self.demotions)),
            ]
        return blocks

    def measurements(
        self, schema_path: str, schema: Schema, commit: str, hardware: str
    ) -> dict[str, Evaluation] | None:
        """Each workload's evaluation by axes text; None if a fault or an unmeasured workload voids them."""
        if any(
            f.schema_path == schema_path and f.code != FindingCode.NO_QUALIFYING_WORKLOAD
            for f in self.findings
        ):
            return None
        records = {axes_text(r.axes): r for r in self.workload_records if r.schema_path == schema_path}
        failed = {
            axes_text(f.axes)
            for f in self.demotions
            if f.schema_path == schema_path and f.axes is not None and _MEASURED_AS[f.code] == "failed"
        }
        measured: dict[str, Evaluation] = {}
        for workload in schema.workloads:
            key = axes_text(workload.axes)
            record = records.get(key)
            if record is None:
                return None
            accuracy = "failed" if key in failed else "passed"
            if accuracy == "passed" and (record.speedup is None or not record.is_compared):
                return None
            speedup = {}
            if record.backend is not None and record.is_timed and record.speedup is not None:
                speedup = {record.backend: Speedup(commit=commit, value=record.speedup)}
            measured[key] = Evaluation(hardware=hardware, accuracy=accuracy, speedup=speedup)
        return measured
