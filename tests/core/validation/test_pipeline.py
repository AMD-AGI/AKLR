# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import io
import json
import math
import os
import re
import statistics
import sys
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import axes_text
from aklr.core.runtime import seam
from aklr.core.validation import (
    AuditStage,
    BenchStage,
    Pipeline,
    PipelineReport,
    ValidationScope,
)
from aklr.core.validation.pipeline import PlatformRecord
from aklr.core.validation.stages import BenchReport, StageReport
from aklr.core.validation.stages._audit._report import AuditReport
from aklr.core.validation.stages._bench._evaluation import (
    CandidateResult,
    CandidateWorkloadResult,
    ReferenceResult,
)
from aklr.core.validation.stages._bench._execution import (
    Executor,
    TaskExecutionFailed,
)
from aklr.core.validation.stages._bench._report import WorkloadRecord
from aklr.core.validation.stages._e2e import _run as _e2e_run
from aklr.core.validation.stages._e2e import _stage as _e2e_stage
from aklr.core.validation.stages._e2e._config import E2eConfig
from aklr.core.validation.stages._e2e._report import E2eRecord, E2eReport
from aklr.core.validation.stages._e2e._run import RunNotKilledError, ScenarioRun
from aklr.core.validation.stages._e2e._stage import E2eStage
from aklr.core.validation.stages._measure import (
    RoundTimes,
    TimingRecord,
    compare_performance,
    gate,
    ratio_lower_bound,
    timing_records,
)
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages._report._markdown import one_line
from aklr.core.validation.stages.stage import StageStatus

# a committed gfx950/MI355X schema, rewritten under tmp_path with a stored speedup
_SCHEMA = "aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json"
_TRITON_SCHEMA = "aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/514a90451e2ca0aa.json"
_PLATFORM = dict.fromkeys(PlatformRecord.model_fields, "unknown") | {
    "arch": "gfx950",
    "hardware": "MI355X",
    "n_visible_devices": 1,
}
_HALVES = [0.0] * 16 + [0.2] * 16  # 32 rounds: df 31, mean 0.1
# the batch label, with the ``_`` that report.md escapes
_LABEL = "/".join((*Path(_SCHEMA).parts[-3:-1], Path(_SCHEMA).stem[:8]))
_LINE = re.compile(r"^\[\d\d:\d\d:\d\d\] (.+)$", re.MULTILINE)


class _FixedStatus:
    def __init__(self, status: StageStatus | None) -> None:
        self.status = status

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> StageReport:
        if self.status is None:
            raise RuntimeError("a fixed stage that raises")
        return self.report_type(status=self.status)


class _FixedAudit(_FixedStatus, AuditStage):
    pass


class _FixedBench(_FixedStatus, BenchStage):
    pass


def _timing(mode: str, versus: str, threshold: str, ratio: float) -> TimingRecord:
    """Two rounds at a constant ``ratio``: its lower bound is ``ratio`` itself."""
    return TimingRecord(
        mode=mode,
        versus=versus,
        threshold=threshold,
        n_rounds=2,
        log_ratio_mean=math.log(ratio),
        ratio_lower_bound=ratio,
        min_ratio=ratio,
        versus_milliseconds=(ratio, ratio),
        candidate_milliseconds=(1.0, 1.0),
    )


def _gate(entry: float, retain: float | None = None) -> FindingCode | None:
    """``gate`` of graph vs baseline at ``entry``, and vs incumbent at ``retain``."""
    records = [_timing("graph", "baseline", "entry", entry)]
    if retain is not None:
        records.append(_timing("graph", "incumbent", "retain", retain))
    return gate(records)


def _workload(
    axes: dict[str, int],
    speedup: float | None,
    fraction: float | None = None,
) -> WorkloadRecord:
    return WorkloadRecord(
        schema_path=_SCHEMA,
        axes=axes,
        kernel_content_hash="",
        baseline="aiter.baseline",
        is_timed=True,
        worst_violation_fraction=fraction,
        timings=(() if speedup is None else (_timing("graph", "baseline", "entry", speedup),)),
        batch_stopped=False,
    )


_TRACER_FAULT = "x: timing: Kineto recorded no kernels for a single call"


def _reruns(*details: str) -> tuple[int, set[FindingCode]]:
    """``_run_batch``'s candidate submits and finding codes, its candidate task failing
    with each of ``details`` in turn."""
    errors = iter(TaskExecutionFailed(d) for d in details)

    def submit(*args: Any, **kwargs: Any) -> Mock:
        error = next(errors)
        return Mock(
            done=lambda: True,
            cancelled=lambda: False,
            exception=lambda: error,
            result=Mock(side_effect=error),
            directory=None,
        )

    executor = Mock(submit=Mock(side_effect=submit))
    scope = ValidationScope(Path.cwd(), "", "HEAD", (), ())
    stage = BenchStage(["cpu"])
    batch, *_ = stage._workloads_to_run(scope, Path(_SCHEMA), ContractRegistry.default())
    findings, _ = stage._run_batch(scope, executor, batch, iter(range(1, 9)), 1)
    return executor.submit.call_count, {f.code for f in findings}


_NONE = ", findings: 0, 0:00:00"
_STARTED_PASSED = [
    "audit: started (stage 1/2)",
    "audit: passed" + _NONE,
    "bench: started (stage 2/2)",
]
_NEWLY_WRONG = [(FindingCode.NUMERIC_MISMATCH, {"m": 1}, "out 0\n  bad")]

_E2E_PASSED = E2eReport(status="passed")
_E2E_PLATFORM_FAULT = E2eReport.from_findings([Finding.unlocated(FindingCode.PLATFORM_FAILURE, "")])


class TestPipeline:
    """``Pipeline``: stages in order behind the audit, the rest skipped after one that does not pass."""

    @pytest.fixture
    def scope(self, tmp_path: Path) -> ValidationScope:
        """A scope rooted at ``tmp_path`` with one schema path, no base."""
        return ValidationScope(tmp_path, "", None, (tmp_path / "schema.json",), ())

    @staticmethod
    def _bench(
        findings: list[tuple[FindingCode, dict[str, int] | None, str]],
        *records: WorkloadRecord,
    ) -> BenchReport:
        """A bench report of ``records`` and ``(code, axes, detail)`` findings on ``_SCHEMA``."""
        return BenchReport.from_findings(
            [Finding(code=c, schema_path=_SCHEMA, axes=a, detail=d) for c, a, d in findings],
            workload_records=records,
        )

    @staticmethod
    def _passed_bench() -> BenchReport:
        """A passed bench report of two workloads."""
        return TestPipeline._bench(
            [],
            _workload({"m": 1}, 1.2, 0.1),
            _workload({"m": 2}, 1.5, 0.1),
        )

    @staticmethod
    def _demotion(code: FindingCode) -> tuple[str, int, int]:
        """A bench report of one workload and ``code`` on it: status, deciding, demoted."""
        report = TestPipeline._bench([(code, {"m": 1}, "")], _workload({"m": 1}, 1.5, 0.0))
        return report.status, len(report.findings), len(report.demotions)

    @staticmethod
    def _writes_back(
        tmp_path: Path,
        speedup: float,
        bench_finding: FindingCode | None,
        e2e: StageReport,
    ) -> bool:
        """Whether ``write_schemas`` rewrites the schema stored at speedup 1.5, measured at
        ``speedup``, with ``bench_finding`` on the schema and ``e2e`` after the bench; an
        earlier run's rewrite in the directory does not count."""
        registry = ContractRegistry.default()
        schema = registry.parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        stored = {
            axes_text(w.axes): w.model_copy(update={"accuracy": "passed", "speedup": 1.5})
            for w in schema.workloads
        }
        for path in (tmp_path / _SCHEMA, tmp_path / "out" / _SCHEMA):
            path.parent.mkdir(parents=True)
            path.write_text(schema.with_measurements(stored).schema_text())
        findings = (
            []
            if bench_finding is None
            else [Finding(code=bench_finding, schema_path=_SCHEMA, axes=None, detail="")]
        )
        bench = BenchReport.from_findings(
            findings,
            workload_records=tuple(_workload(w.axes, speedup, 0.0) for w in schema.workloads),
        )
        PipelineReport(
            head_sha="",
            base_sha=None,
            measurement_profile={},
            platform=PlatformRecord(**_PLATFORM),
            schema_paths=(_SCHEMA,),
            bench_paths=(_SCHEMA,),
            stages=(AuditReport(status="passed"), bench, e2e),
        ).write_schemas(tmp_path, tmp_path / "out")
        return (tmp_path / "out" / _SCHEMA).exists()

    def test_rejects_bad_stage_order(self) -> None:
        """``[bench, audit]`` (no audit first) and ``[audit, audit]``."""
        with pytest.raises(ValueError, match="first stage"):
            Pipeline([_FixedBench("passed"), _FixedAudit("passed")])
        with pytest.raises(ValueError, match="not in the order"):
            Pipeline([_FixedAudit("passed"), _FixedAudit("passed")])

    @pytest.mark.parametrize(
        ("audit", "bench", "exit_code", "progress"),
        [
            ("passed", "passed", 0, [*_STARTED_PASSED, "bench: passed" + _NONE]),
            (
                "failed",
                "skipped",
                1,
                [
                    "audit: started (stage 1/2)",
                    "audit: failed" + _NONE,
                    "bench: skipped: the previous stage audit failed",
                ],
            ),
            ("passed", "failed", 1, [*_STARTED_PASSED, "bench: failed" + _NONE]),
        ],
    )
    def test_run_skips_after_failure(
        self,
        scope: ValidationScope,
        capsys: pytest.CaptureFixture[str],
        audit: StageStatus,
        bench: StageStatus,
        exit_code: int,
        progress: list[str],
    ) -> None:
        # a bench to be skipped raises: running it would report it aborted
        fixed_bench = _FixedBench(None if bench == "skipped" else bench)
        report = Pipeline([_FixedAudit(audit), fixed_bench]).run(scope)
        assert [s.status for s in report.stages] == [audit, bench]
        assert report.exit_code == exit_code
        lines = _LINE.findall(capsys.readouterr().err)
        assert lines[0].startswith("validate: ") and ", head " in lines[0]
        assert lines[1:] == progress

    @pytest.mark.parametrize(
        ("base_path", "first_release", "threshold"),
        [
            ("aklr/ops/gemm/impls/triton/fp8/fp8_gfx950_m1.json", True, "retain"),
            (_TRITON_SCHEMA, False, "entry"),
        ],
        ids=["legacy_base", "schema_base"],
    )
    def test_first_release_uses_retain_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        base_path: str,
        first_release: bool,
        threshold: str,
    ) -> None:
        """A base holding no schema data uses the retain timing threshold."""
        monkeypatch.setattr(ValidationScope, "paths_at_base", lambda self, directory: (base_path,))
        scope = ValidationScope(Path.cwd(), "", "base", (), ())
        assert scope.is_first_release is first_release
        stage = BenchStage(["cpu"])
        batch, *_ = stage._workloads_to_run(scope, Path(_SCHEMA), ContractRegistry.default())
        assert {r.thresholds["baseline"] for r in batch} == {threshold}

    def test_no_schema_skips_every_stage(self, scope: ValidationScope) -> None:
        """A scope with no schema path: every stage ``skipped``, exit code 0."""
        pipeline = Pipeline([_FixedAudit("passed"), _FixedBench("passed")])
        report = pipeline.run(replace(scope, schema_paths=()))
        assert [s.status for s in report.stages] == ["skipped", "skipped"]
        assert report.exit_code == 0

    def test_raising_stage_is_aborted(
        self, scope: ValidationScope, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bench that raises is reported ``aborted`` ("internal: ..."), the report is still made."""
        report = Pipeline([_FixedAudit("passed"), _FixedBench(None)]).run(scope)
        bench = report.stages[1]
        assert bench.status == "aborted"
        assert bench.detail.startswith("internal: ")
        assert report.exit_code == 3
        assert (
            "bench: aborted (internal: RuntimeError: a fixed stage that raises),"
            " findings: 0, 0:00:00" in _LINE.findall(capsys.readouterr().err)
        )

    @pytest.mark.parametrize(
        ("kill", "logged"),
        [
            pytest.param(
                lambda m: m.setattr(sys, "stderr", io.StringIO()) or sys.stderr.close(),
                False,
                id="closed-stderr",
            ),
            pytest.param(lambda m: m.setattr(sys, "stderr", None), False, id="no-stderr"),
            pytest.param(
                lambda m: m.setattr(sys, "stderr", Mock(**{"write.side_effect": BrokenPipeError})),
                False,
                id="broken-pipe",
            ),
            pytest.param(
                lambda m: m.setattr(
                    StageReport,
                    "console_lines",
                    Mock(side_effect=RuntimeError("text")),
                ),
                True,
                id="failing-text",
            ),
        ],
    )
    def test_dead_progress_keeps_report(
        self,
        scope: ValidationScope,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        kill: Callable[[pytest.MonkeyPatch], None],
        logged: bool,
    ) -> None:
        """A closed, missing or broken stderr, or a failing finish text, costs no report."""
        kill(monkeypatch)
        report = Pipeline([_FixedAudit("passed"), _FixedBench("failed")]).run(scope)
        assert [s.status for s in report.stages] == ["passed", "failed"]
        assert report.exit_code == 1
        assert ("progress text failed" in caplog.text) == logged

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            # _measure: gate against entry (1.02) and retain (0.99), the t lower bound
            pytest.param(lambda _: _gate(1.05), None, id="gate-faster"),
            pytest.param(lambda _: _gate(1.0), FindingCode.NOT_FASTER, id="gate-equal"),
            pytest.param(lambda _: _gate(1.05, 0.98), FindingCode.REGRESSED, id="gate-regressed"),
            pytest.param(lambda _: _gate(0.5, 0.5), FindingCode.REGRESSED, id="gate-both-below"),
            pytest.param(lambda _: ratio_lower_bound([0.1]), ValueError, id="bound-one-round"),
            pytest.param(
                lambda _: ratio_lower_bound([math.log(1.5)] * 2),
                pytest.approx(1.5),
                id="bound-constant",
            ),
            pytest.param(
                lambda _: ratio_lower_bound(_HALVES),
                pytest.approx(
                    math.exp(0.1 - 1.6955 * statistics.stdev(_HALVES) / math.sqrt(32)),
                    rel=1e-4,
                ),
                id="bound-t-df31",
            ),
            # BenchReport.from_findings: (status, deciding findings, demotions)
            pytest.param(
                lambda _: TestPipeline._demotion(FindingCode.NUMERIC_MISMATCH),
                ("failed", 1, 1),
                id="no-correct-workload",
            ),
            pytest.param(
                lambda _: TestPipeline._demotion(FindingCode.NOT_FASTER),
                ("passed", 0, 1),
                id="not-faster-demotes",
            ),
            pytest.param(
                lambda _: TestPipeline._demotion(FindingCode.REGRESSED),
                ("passed", 0, 1),
                id="regressed-demotes",
            ),
            # PipelineReport.write_schemas: whether the schema stored at 1.5 is rewritten
            pytest.param(
                lambda p: TestPipeline._writes_back(p, 1.56, None, _E2E_PASSED),
                True,
                id="moved-4%",
            ),
            pytest.param(
                lambda p: TestPipeline._writes_back(p, 1.52, None, _E2E_PASSED),
                False,
                id="moved-1.3%",
            ),
            pytest.param(
                lambda p: TestPipeline._writes_back(p, 1.56, None, _E2E_PLATFORM_FAULT),
                True,
                id="e2e-platform-fault",
            ),
            pytest.param(
                lambda p: TestPipeline._writes_back(p, 1.56, FindingCode.PLATFORM_FAILURE, _E2E_PASSED),
                False,
                id="bench-aborted",
            ),
            # console text: one_line, plain_item, console_lines, batch_line, outcome_lines
            pytest.param(lambda _: one_line("ab " * 100, 5), "ab ab...", id="one-line-cut"),
            pytest.param(
                lambda _: Finding(
                    code=FindingCode.SCHEMA_INVALID,
                    schema_path="a/b.json",
                    axes={"m": 1},
                    detail=" x\n y ",
                ).plain_item(),
                "schema_invalid a/b.json: x y",
                id="item-located",
            ),
            pytest.param(
                lambda _: Finding.unlocated(FindingCode.PLATFORM_FAILURE, "a\n b").plain_item(),
                "platform_failure: a b",
                id="item-unlocated",
            ),
            pytest.param(
                lambda _: AuditReport.from_findings(
                    [Finding.unlocated(FindingCode.SCHEMA_INVALID, str(i)) for i in range(12)]
                ).console_lines(),
                [
                    "failed, findings: 12, 0:00:00",
                    *(f"  schema_invalid: {i}" for i in range(10)),
                    "  ... and 2 more, see report.md",
                ],
                id="finish-overflow",
            ),
            pytest.param(
                lambda _: (
                    AuditReport.aborted("internal: X\n Y")
                    .model_copy(update={"elapsed_seconds": 206.4})
                    .console_lines()
                ),
                ["aborted (internal: X Y), findings: 0, 0:03:26"],
                id="finish-aborted",
            ),
            pytest.param(
                lambda _: BenchReport.skipped("no kernel to measure").console_lines(),
                ["skipped: no kernel to measure"],
                id="finish-skipped",
            ),
            pytest.param(
                lambda _: E2eRecord(
                    scenario="s",
                    n_runs_per_side=2,
                    throughput_ratio_lower_bound=1.02,
                    latency_ratio_lower_bound=1.05,
                    n_hit_calls=3,
                    n_fallback_calls=0,
                ).console_line(1.01),
                "throughput ratio lower bound 1.0200, gate >= 1.0100,"
                " latency ratio lower bound 1.0500, 3 hit calls, 0 fallback calls",
                id="e2e-line",
            ),
            pytest.param(
                lambda _: TestPipeline._passed_bench().batch_line(),
                f"passed {_LABEL}: 2 wl, speedup 1.20x-1.50x, acc ok",
                id="line-passed",
            ),
            pytest.param(
                lambda _: TestPipeline._bench([], _workload({"m": 1}, 1.5)).batch_line(),
                f"failed {_LABEL}: 1 wl, speedup 1.50x, acc -",
                id="line-equal-speedup",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(
                    [(FindingCode.BASELINE_FAILED, {"m": 2}, "declined")],
                    _workload({"m": 1}, 1.5, 0.1),
                    _workload({"m": 2}, None),
                ).batch_line(),
                f"failed {_LABEL}: 2 wl, speedup 1.50x, acc ok 1/2",
                id="line-partial-acc",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(
                    [(FindingCode.REGRESSED, {"m": 1}, "")],
                    _workload({"m": 1}, 1.5),
                ).batch_line(),
                f"failed {_LABEL}: 1 wl, speedup 1.50x, acc -",
                id="line-demoted-only",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(_NEWLY_WRONG, _workload({"m": 1}, 1.5, 3.1)).batch_line(),
                f"failed {_LABEL}: 1 wl, speedup 1.50x, acc FAIL 1/1",
                id="line-newly-wrong",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(
                    [(FindingCode.CANDIDATE_CRASH, {"m": k}, "") for k in (1, 2)],
                    *(_workload({"m": k}, None).model_copy(update={"batch_stopped": True}) for k in (1, 2)),
                ).batch_line(),
                f"failed {_LABEL}: 2 wl, acc -",
                id="line-stopped-crash",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(
                    [
                        (
                            FindingCode.PLATFORM_FAILURE,
                            None,
                            "reference: RunnerStartupFailed",
                        )
                    ],
                    _workload({"m": 1}, 1.5),
                ).batch_line(),
                f"aborted {_LABEL}: 1 wl, speedup 1.50x, acc -",
                id="line-unlocated",
            ),
            pytest.param(
                lambda _: BenchReport(status="passed").outcome_lines(),
                [],
                id="outcome-empty",
            ),
            pytest.param(
                lambda _: TestPipeline._bench(
                    [(FindingCode.NOT_FASTER, {"m": 2}, "")],
                    _workload({"m": 1}, 1.2, 0.1),
                    _workload({"m": 2}, 2.0),
                ).outcome_lines(),
                [
                    "2 workloads",
                    f"  {'outcome':<32}{'workloads':>9}",
                    f"  {'passed':<32}{1:>9}",
                    f"  {'not_faster (demoted)':<32}{1:>9}",
                    ("  speedup  best 2.00x, median 1.60x, worst 1.20x  (workloads with a speedup value)"),
                    "  accuracy FAIL 0, ok 1, not measured 1",
                    "  per-schema table: report.md",
                ],
                id="outcome",
            ),
            pytest.param(
                lambda _: _reruns(_TRACER_FAULT, _TRACER_FAULT),
                (2, {FindingCode.PLATFORM_FAILURE}),
                id="tracer-reruns-once",
            ),
            pytest.param(
                lambda _: _reruns("boom"),
                (1, {FindingCode.CANDIDATE_CRASH}),
                id="crash-no-rerun",
            ),
            pytest.param(
                lambda _: TestPipeline._passed_bench().console_lines()[-1],
                "passed, findings: 0, 0:00:00",
                id="bench-console",
            ),
            pytest.param(
                lambda _: [
                    ValidationScope.path_kind(p) for p in (".github/README.md", ".github/workflows/ci.yaml")
                ],
                ["document", "other"],
                id="github-md-is-document",
            ),
        ],
    )
    def test_verdict_rules(self, tmp_path: Path, case: Callable[[Path], Any], expected: Any) -> None:
        """The bench gate and bound, which findings decide the bench, and when a
        measured schema is written back, and the console text of reports."""
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                case(tmp_path)
        else:
            assert case(tmp_path) == expected


class TestSpeedupMeasures:
    """The reported ratios are unclamped; the speedup is the min over the modes."""

    def test_unclamped_mean_clamped_gate_bound(self) -> None:
        """A constant 3x: mean and min ratio 3, the gate's lower bound at most 2."""
        rounds = [
            RoundTimes(
                mode="graph",
                round_index=i,
                order=("baseline", "candidate"),
                milliseconds={"baseline": 3.0, "candidate": 1.0 + 1e-3 * i},
            )
            for i in range(4)
        ]
        record = timing_records(rounds, {"baseline": "entry"})[0]
        assert math.exp(record.log_ratio_mean) == pytest.approx(3.0, rel=1e-2)
        assert record.min_ratio == pytest.approx(3.0, rel=1e-2)
        assert record.ratio_lower_bound <= 2.0

    def test_non_finite_time_is_a_candidate_crash(self) -> None:
        """A NaN round time, which the runner's json carries, is a finding."""
        rounds = [
            RoundTimes(
                mode="graph",
                round_index=i,
                order=("baseline", "candidate"),
                milliseconds={
                    "baseline": 1.0,
                    "candidate": math.nan if i == 0 else 1.0,
                },
            )
            for i in range(32)
        ]
        records, finding = compare_performance(rounds, {"baseline": "entry"})
        assert records == ()
        assert finding is not None and finding.code is FindingCode.CANDIDATE_CRASH

    def test_speedup_is_the_graph_bound(self) -> None:
        """Graph 1.5 vs the baseline: 1.5."""
        record = _workload({"m": 1}, 1.5)
        assert record.speedup == pytest.approx(1.5)

    def test_one_correct_fast_workload_admits_kernel(self) -> None:
        report = TestPipeline._bench(
            [(FindingCode.POST_TIMING_MISMATCH, {"m": 1}, "stale weights")],
            _workload({"m": 1}, 2.0, 1.0),
            _workload({"m": 2}, 1.5, 0.0),
        )
        assert report.is_passed
        assert report.accuracy() == (1, 1)
        assert len(report.demotions) == 1

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.99, False), (1.0, False), (1.02, False), (1.0200001, True)],
    )
    def test_same_workload_must_be_correct_and_fast(self, ratio: float, expected: bool) -> None:
        report = TestPipeline._bench(
            [(FindingCode.POST_TIMING_MISMATCH, {"m": 1}, "")],
            _workload({"m": 1}, 3.0, 1.0),
            _workload({"m": 2}, ratio, 0.0),
        )
        assert report.is_passed is expected
        assert [f.code for f in report.findings] == ([] if expected else [FindingCode.NO_QUALIFYING_WORKLOAD])

    def test_every_kernel_needs_a_qualifying_workload(self) -> None:
        report = BenchReport.from_findings(
            [],
            workload_records=(
                _workload({"m": 1}, 1.5, 0.0),
                _workload({"m": 1}, 0.99, 0.0).model_copy(update={"schema_path": _TRITON_SCHEMA}),
            ),
        )
        assert report.status == "failed"
        assert [(f.code, f.schema_path) for f in report.findings] == [
            (FindingCode.NO_QUALIFYING_WORKLOAD, _TRITON_SCHEMA)
        ]

    def test_untimed_correct_workload_without_baseline_qualifies(self) -> None:
        record = _workload({"m": 1}, None, 0.0).model_copy(update={"is_timed": False, "baseline": None})
        assert TestPipeline._bench([], record).is_passed

    @pytest.mark.parametrize("qualifying_ratio", [0.99, 1.5])
    def test_admission_preserves_failed_workload_dispatch_exclusion(self, qualifying_ratio: float) -> None:
        schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        failed_axes = schema.workloads[0].axes
        records = tuple(_workload(w.axes, qualifying_ratio, 0.0) for w in schema.workloads)
        report = TestPipeline._bench([(FindingCode.POST_TIMING_MISMATCH, failed_axes, "")], *records)
        assert report.is_passed is (qualifying_ratio > 1.02)
        measured = report.measurements(_SCHEMA, schema)
        assert measured is not None
        failed = measured[axes_text(failed_axes)]
        assert failed.accuracy == "failed"
        assert not failed.qualifies(0)
        for w in schema.workloads[1:]:
            assert measured[axes_text(w.axes)].accuracy == "passed"

    @pytest.mark.parametrize(
        "code",
        [
            FindingCode.BUILD_FAILED,
            FindingCode.CANDIDATE_CRASH,
            FindingCode.CANDIDATE_TIMEOUT,
        ],
    )
    @pytest.mark.parametrize("has_qualifying_workload", [False, True])
    def test_candidate_batch_failure_only_excludes_its_workloads(
        self, code: FindingCode, has_qualifying_workload: bool
    ) -> None:
        schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        records = tuple(
            (
                _workload(w.axes, 1.5, 0.0)
                if index == 0 and has_qualifying_workload
                else _workload(w.axes, None).model_copy(update={"batch_stopped": True})
            )
            for index, w in enumerate(schema.workloads)
        )
        report = TestPipeline._bench(
            [(code, r.axes, "batch failed") for r in records if r.batch_stopped],
            *records,
        )
        assert report.is_passed is has_qualifying_workload
        assert [f.code for f in report.findings] == (
            [] if has_qualifying_workload else [FindingCode.NO_QUALIFYING_WORKLOAD]
        )
        measured = report.measurements(_SCHEMA, schema)
        assert measured is not None
        for record in records:
            workload = measured[axes_text(record.axes)]
            assert workload.qualifies(0) is not record.batch_stopped
            if record.batch_stopped:
                assert workload.accuracy == "failed"
                assert workload.speedup is None
            assert not {"is_kept", "known_failed"} & record.model_dump().keys()
        assert PipelineReport.model_fields["version"].default == 2

    def test_platform_batch_failure_still_blocks_qualifying_kernel(self) -> None:
        report = TestPipeline._bench(
            [(FindingCode.PLATFORM_FAILURE, {"m": 2}, "reference worker died")],
            _workload({"m": 1}, 1.5, 0.0),
            _workload({"m": 2}, None).model_copy(update={"batch_stopped": True}),
        )
        assert report.status == "aborted"
        assert [f.code for f in report.findings] == [FindingCode.PLATFORM_FAILURE]
        schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        assert report.measurements(_SCHEMA, schema) is None


def _custom_baseline(a, b, trans_b=True, bias=None, out_dtype=None, out=None):
    return a @ b.T


class TestBaseline:
    """What a workload is compared with: a base kernel dispatch serves (it beats the
    defined baseline), else the defined baseline, else nothing (the new case is its
    own)."""

    _A, _B, *_ = (axes_text(w["axes"]) for w in json.loads(Path(_SCHEMA).read_text())["workloads"])

    @staticmethod
    def _requests(scope: ValidationScope) -> dict[str, Any]:
        batch, *_ = BenchStage(["cpu"])._workloads_to_run(scope, Path(_SCHEMA), ContractRegistry.default())
        return {r.axes_text: r for r in batch}

    @staticmethod
    def _base_text(
        speedups: dict[str, float],
        same_kernel: bool = False,
        author: str = "",
    ) -> str:
        """``_SCHEMA`` with the selected kernel, author and measured workloads."""
        schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        first, *rest = schema.kernel.sources
        content = first.content + ("" if same_kernel else "\n# the base's kernel\n")
        kernel = schema.kernel.model_copy(
            update={
                "sources": (first.model_copy(update={"content": content}), *rest),
                "author": author,
            }
        )
        workloads = tuple(
            w.model_copy(update={"accuracy": "passed", "speedup": speedups[axes_text(w.axes)]})
            for w in schema.workloads
            if axes_text(w.axes) in speedups
        )
        return schema.model_copy(update={"kernel": kernel, "workloads": workloads}).schema_text()

    @staticmethod
    def _base_scope(
        monkeypatch: pytest.MonkeyPatch,
        speedups: dict[str, float],
        same_kernel: bool = False,
    ) -> ValidationScope:
        """A base holding ``_SCHEMA`` at the selected workloads and speedups."""
        text = TestBaseline._base_text(speedups, same_kernel)
        monkeypatch.setattr(ValidationScope, "paths_at_base", lambda self, d: (_SCHEMA,))
        monkeypatch.setattr(ValidationScope, "text_at_base", lambda self, path: text)
        return ValidationScope(Path.cwd(), "", "base", (), ())

    def test_unreadable_incumbent_is_a_platform_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A base schema of the same name that does not parse: the schema is never
        timed without it, a located PLATFORM_FAILURE; the bench goes on."""
        scope = replace(self._base_scope(monkeypatch, {self._A: 1.2}), bench_paths=(Path(_SCHEMA),))
        monkeypatch.setattr(ValidationScope, "text_at_base", lambda self, path: "{")
        monkeypatch.setattr(Executor, "for_devices", lambda devices: nullcontext())
        report = BenchStage(["cpu"]).run(scope, None)
        assert [(f.code, f.schema_path) for f in report.findings] == [(FindingCode.PLATFORM_FAILURE, _SCHEMA)]
        assert "cannot read the incumbent" in report.findings[0].detail

    @pytest.mark.parametrize("contract_id", ["gone@1.0", None])
    def test_base_of_a_rewritten_contract_is_no_incumbent(
        self, monkeypatch: pytest.MonkeyPatch, contract_id: str | None
    ) -> None:
        """A base schema that is JSON but not this contract's schema text (its contract
        was removed or rewritten here): no incumbent, timed against the defined baseline."""
        base = json.loads(self._base_text({self._A: 1.2}))
        if contract_id:
            base["contract_id"] = contract_id
        else:
            base["unknown_key"] = 1
        scope = self._base_scope(monkeypatch, {self._A: 1.2})
        monkeypatch.setattr(ValidationScope, "text_at_base", lambda self, path: json.dumps(base))
        assert all(r.incumbent is None for r in self._requests(scope).values())

    def test_record_names_the_route_that_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Of several routes the first that binds is timed, so it names the defined
        baseline; a served base kernel stays the baseline; a workload stopped before
        timing has none to export."""
        scope = self._base_scope(monkeypatch, {self._A: 1.2})
        requests = self._requests(scope)
        bound = "aiter.ops.the_route_that_bound"
        crash = Finding.unlocated(FindingCode.CANDIDATE_CRASH, "")
        candidate = CandidateResult(
            findings=(),
            workload_results=tuple(
                CandidateWorkloadResult(
                    axes_text=r.axes_text,
                    findings=() if r.axes_text == self._B else (crash,),
                    rounds=(),
                    baseline=bound if r.axes_text == self._B else None,
                )
                for r in requests.values()
            ),
        )
        reference = ReferenceResult(workload_results=())
        _, records = BenchStage(["cpu"])._judged_findings(
            scope, tuple(requests.values()), [], candidate, reference
        )
        baselines = {axes_text(r.axes): r.baseline for r in records}
        assert baselines.pop(self._A) == _SCHEMA
        assert baselines.pop(self._B) == bound
        assert set(baselines.values()) == {None}

    def test_custom_baseline_without_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``baseline.py`` function and no route: the defined baseline, timed."""
        monkeypatch.setattr(seam, "_ROUTES", {})
        monkeypatch.setattr(seam, "custom_baseline", lambda contract: _custom_baseline)
        requests = self._requests(ValidationScope(Path.cwd(), "", None, (), ()))
        name = f"{_custom_baseline.__module__}.{_custom_baseline.__qualname__}"
        assert {r.baseline for r in requests.values()} == {name}
        assert all("baseline" in r.thresholds for r in requests.values())

    def test_incumbent_only_when_it_beats_the_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A at 1.2 is retained against the base kernel; B at 0.9 is compared with the
        defined baseline alone."""
        requests = self._requests(self._base_scope(monkeypatch, {self._A: 1.2, self._B: 0.9}))
        contract = ContractRegistry.default().get(requests[self._A].contract.contract_id)
        assert requests[self._A].thresholds["incumbent"] == "retain"
        assert requests[self._A].baseline == _SCHEMA
        assert requests[self._B].incumbent is None
        assert requests[self._B].baseline == seam.baseline_of(contract) is not None

    def test_no_defined_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No defined baseline: a base kernel at 1.0 is the baseline (retain); with
        none, nothing is timed and the speedup is 1.0."""
        monkeypatch.setattr(seam, "baseline_of", lambda contract: None)
        requests = self._requests(self._base_scope(monkeypatch, {self._A: 1.0}))
        assert requests[self._A].thresholds == {"baseline": "retain"}
        assert requests[self._A].baseline == _SCHEMA
        assert requests[self._B].thresholds == {}
        assert requests[self._B].baseline is None
        assert compare_performance((), {}) == ((), None)
        record = _workload({"m": 1}, None).model_copy(update={"baseline": None, "is_timed": False})
        assert record.speedup == 1.0

    @pytest.mark.parametrize("has_defined_baseline", [True, False])
    def test_served_kernel_itself_is_the_baseline(
        self, monkeypatch: pytest.MonkeyPatch, has_defined_baseline: bool
    ) -> None:
        """This very kernel, served at base, is the exported baseline yet never timed
        against itself: A at 1.2 always; B at 0.9 only without a defined baseline."""
        if not has_defined_baseline:
            monkeypatch.setattr(seam, "baseline_of", lambda contract: None)
        scope = self._base_scope(monkeypatch, {self._A: 1.2, self._B: 0.9}, same_kernel=True)
        requests = self._requests(scope)
        a, b = requests[self._A], requests[self._B]
        assert a.baseline == _SCHEMA and a.incumbent is None
        assert "incumbent" not in a.thresholds
        assert ("baseline" in a.thresholds) is has_defined_baseline
        # min_speedup 0 without a defined baseline: dispatch serves B at 0.9 too
        assert (b.baseline == _SCHEMA) is not has_defined_baseline
        crash = Finding.unlocated(FindingCode.CANDIDATE_CRASH, "")
        _, (record,) = BenchStage(["cpu"])._judged_findings(scope, (a,), [crash], None, None)
        assert record.baseline == _SCHEMA
        assert record.is_timed is has_defined_baseline

    def test_author_only_edit_is_this_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A base file of this very kernel but another ``author`` is this kernel: kept
        at both workloads (the retention threshold, B at 0.9 included), never timed
        against itself, and chosen over a faster base kernel at A."""
        other = Path(_SCHEMA).with_name("0123456789abcdef.json").as_posix()
        texts = {
            _SCHEMA: self._base_text({self._A: 1.2, self._B: 0.9}, same_kernel=True, author="TEST_author"),
            other: self._base_text({self._A: 1.5}),
        }
        monkeypatch.setattr(ValidationScope, "paths_at_base", lambda self, d: (*texts,))
        monkeypatch.setattr(ValidationScope, "text_at_base", lambda self, p: texts[p])
        requests = self._requests(ValidationScope(Path.cwd(), "", "base", (), ()))
        a, b = requests[self._A], requests[self._B]
        assert a.thresholds == b.thresholds == {"baseline": "retain"}
        assert a.incumbent is None and b.incumbent is None
        assert a.baseline == _SCHEMA

    def test_write_baselines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Each run replaces the export: the base kernel's schema for its workloads,
        the defined baseline in main's Solution shape for the rest, nothing for a
        workload with no baseline."""
        scope = self._base_scope(monkeypatch, {self._A: 1.2})
        schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        route = seam.baseline_of(ContractRegistry.default().get(schema.contract_id))
        out = tmp_path / "baseline"
        directory = out / schema.op_type / schema.name
        solution = directory / f"{schema.name}__baseline.json"

        def write(baselines: dict[str, str | None]) -> list[str]:
            records = tuple(
                _workload(w.axes, 1.1).model_copy(update={"baseline": baselines.get(axes_text(w.axes))})
                for w in schema.workloads
            )
            PipelineReport(
                head_sha="",
                base_sha="base",
                measurement_profile={},
                platform=PlatformRecord(**_PLATFORM),
                schema_paths=(_SCHEMA,),
                bench_paths=(_SCHEMA,),
                stages=(
                    AuditReport(status="passed"),
                    BenchReport.from_findings([], workload_records=records),
                ),
            ).write_baselines(scope, out)
            return sorted(p.name for p in out.rglob("*.json"))

        base_file = Path(_SCHEMA).name
        assert write({self._A: _SCHEMA, self._B: route}) == sorted([base_file, solution.name])
        workloads_of = lambda p: [axes_text(w["axes"]) for w in json.loads(p.read_text())["workloads"]]
        assert workloads_of(directory / base_file) == [self._A]
        assert workloads_of(solution) == [self._B]
        written = json.loads(solution.read_text())
        assert written["author"] == "baseline"
        assert written["spec"]["entry_point"] == "main.py::run"
        assert "@aklr_route" in written["sources"][0]["content"]
        # B's base kernel now qualifies too: the defined baseline's file goes
        assert write({self._A: _SCHEMA, self._B: _SCHEMA}) == [base_file]
        # a custom baseline.py function is exported as main.py
        monkeypatch.setattr(seam, "custom_baseline", lambda c: _custom_baseline)
        assert write({self._A: None, self._B: "x._custom_baseline"}) == [solution.name]
        assert "def _custom_baseline" in json.loads(solution.read_text())["sources"][0]["content"]
        assert write({self._A: None, self._B: None}) == []


@pytest.mark.parametrize("cache_key", ["none", "address", "version"])
def test_post_timing_exposes_stale_weight_cache(monkeypatch: pytest.MonkeyPatch, cache_key: str) -> None:
    """Existing post-timing calls expose cached weights even without a version bump."""
    import torch

    from aklr.core.validation.stages._bench._evaluation._candidate import CandidateRun
    from tests.op_case import OpCase

    request = OpCase.of("gemm@1.0").request
    timing_call = request.assemble_call(request.compose_inputs(None), "cpu")
    weight = timing_call["b"]
    pointer, version = weight.data_ptr(), weight._version
    cache = {(pointer, version if cache_key == "version" else 0): weight.clone()}
    graph = Mock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    capture = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", capture)

    def dispatch(call: dict[str, Any]) -> Any:
        b = call["b"]
        if cache_key != "none":
            key = (b.data_ptr(), b._version if cache_key == "version" else 0)
            if key not in cache:
                cache[key] = b.clone()
            b = cache[key]
        out = call["out"]
        if out is None:
            out = torch.empty(
                request.contract.storage_shape_of("out", request.point),
                dtype=call["a"].dtype,
            )
        graph.replay.side_effect = lambda: out.copy_(call["a"] @ b.mT)
        return out

    dispatcher = Mock(dispatch=Mock(side_effect=dispatch))
    tripwire = Mock(armed=lambda: nullcontext([]))
    steps = [s for s in request.steps if s.check == "post_timing"]
    outputs = {}
    CandidateRun("cpu", tripwire, None).produce_outputs(
        request, dispatcher, steps, outputs, timing_call=timing_call
    )

    assert weight.data_ptr() == pointer and weight._version == version
    assert len(steps) == graph.replay.call_count == 4
    assert capture.call_count == dispatcher.dispatch.call_count == 3
    for step in steps:
        values = request.compose_inputs(step)
        expected = values["a"] @ values["b"].mT
        assert torch.equal(outputs[step.output_key("out")], expected) == (cache_key == "none")
        assert step.mismatch_code is FindingCode.POST_TIMING_MISMATCH


def test_a_call_writing_its_input_is_input_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay that writes an input fails its step as INPUT_MUTATION."""
    import torch

    from aklr.core.validation.stages._bench._evaluation._candidate import (
        CandidateRun,
        StepFailed,
    )
    from tests.op_case import OpCase

    request = OpCase.of("gemm@1.0").request
    graph = Mock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **k: nullcontext())

    def dispatch(call: dict[str, Any]) -> Any:
        a = call["a"]
        out = torch.empty(request.contract.storage_shape_of("out", request.point), dtype=a.dtype)

        def replay() -> None:
            out.copy_(a @ call["b"].mT)
            a.view(-1)[0] += 1.0

        graph.replay.side_effect = replay
        return out

    dispatcher = Mock(dispatch=Mock(side_effect=dispatch))
    tripwire = Mock(armed=lambda: nullcontext([]))
    step = next(s for s in request.steps if s.buffer_source == "new")
    with pytest.raises(StepFailed) as failed:
        CandidateRun("cpu", tripwire, None).produce_outputs(request, dispatcher, [step], {})
    assert failed.value.code is FindingCode.INPUT_MUTATION


def test_warm_up_covers_the_workload_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    """mhc's workloads set scalars (post_multiplier 2.0) its kernel compiles per value:
    the warm-up calls them too, so the captured workload step compiles nothing."""
    import torch

    from aklr.core.runtime.builder import BuilderRegistry
    from aklr.core.validation.stages._bench._evaluation import TrialRequest
    from aklr.core.validation.stages._bench._evaluation._candidate import (
        CandidateRun,
    )

    path = str(min(Path("aklr/ops/mhc/impls").rglob("*.json")))
    schema = ContractRegistry.default().parse_schema(Path(path).read_bytes(), path)
    contract = ContractRegistry.default().get(schema.contract_id)
    request = TrialRequest.of_workload(
        path,
        schema,
        contract,
        schema.workloads[0],
        thresholds={},
        incumbent=None,
        baseline=None,
    )
    assert request.has_workload_set
    seen = []
    runnable = Mock(side_effect=lambda **call: seen.append(call["post_multiplier"]))
    monkeypatch.setattr(BuilderRegistry, "build", lambda self, s: runnable)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    CandidateRun("cpu", None, None).make_warmed_dispatcher(schema, [request])
    assert sorted(seen) == [1.0, 2.0]


def test_long_schema_spreads_over_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema of more workloads than a batch holds splits into even batches of at
    most ``MAX_BATCH_WORKLOADS``, every workload once, in schema order."""
    from aklr.core.validation.stages._bench import _stage

    monkeypatch.setattr(_stage, "MAX_BATCH_WORKLOADS", 4)
    scope = ValidationScope(Path.cwd(), "", "HEAD", (), ())
    schema = ContractRegistry.default().parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
    batches = BenchStage(["cpu"])._workloads_to_run(scope, Path(_SCHEMA), ContractRegistry.default())
    assert len(schema.workloads) == 13
    assert [len(b) for b in batches] == [4, 3, 3, 3]
    assert [r.axes_text for b in batches for r in b] == [axes_text(w.axes) for w in schema.workloads]


@pytest.mark.parametrize(
    ("kernel", "code"),
    [
        ({"targets": [{"arch": "gfx942"}]}, FindingCode.NO_MATCHING_DEVICE),
        ({"requires": {"flydsl": ">=999"}}, FindingCode.NO_MATCHING_TOOLCHAIN),
    ],
    ids=["targets", "requires"],
)
def test_platform_the_kernel_does_not_state_aborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kernel: dict[str, Any],
    code: FindingCode,
) -> None:
    """A device or toolchain the kernel does not state is the platform's fault: the
    candidate stops before its kernel is built."""
    from aklr.core.contract.spec import KernelSpec
    from aklr.core.platform.device import DeviceTarget
    from aklr.core.validation.stages._bench._evaluation import run_candidate
    from aklr.core.validation.stages._bench._execution import ExecutionContext
    from tests.op_case import KERNEL_TARGET, OpCase

    case = OpCase.of("gemm@1.0")
    schema = case.schema()
    kernel_spec = KernelSpec.model_validate(schema.kernel.model_dump() | kernel)
    schema = schema.model_copy(update={"kernel": kernel_spec})
    target = DeviceTarget.from_names(**KERNEL_TARGET)
    monkeypatch.setattr(DeviceTarget, "current", classmethod(lambda cls, d: target))
    context = ExecutionContext(device="cpu", log_path=tmp_path / "log", directory=tmp_path)
    result = run_candidate(context, requests=[replace(case.request, schema=schema)])
    findings = CandidateResult.model_validate(result).findings
    assert [f.code for f in findings] == [code]
    assert code.is_platform_fault


def test_importing_validation_loads_no_torch() -> None:
    """Audit-only runs stay light: torch loads only once a bench runs."""
    import subprocess

    code = "import sys, aklr.core.validation; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


class TestE2e:
    @pytest.mark.parametrize("on_ms", [(4.0, 4.0), (8.0, 8.0), (4.0, 8.0)])
    def test_latency_comparison(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        on_ms: tuple[float, float],
    ) -> None:
        """Latency uses paired off/on bounds, independently of throughput."""
        rounds = [
            (
                _e2e_run.RunResult(tokens_per_s=2, latency_ms=ms),
                _e2e_run.RunResult(tokens_per_s=1, latency_ms=6),
            )
            for ms in on_ms
        ]
        monkeypatch.setattr(E2eStage, "_compare_rounds", lambda *args: rounds)
        monkeypatch.setattr(ScenarioRun, "execute", lambda self: None)
        monkeypatch.setattr(_e2e_stage, "read_calls", lambda directory: [])
        scenarios = E2eConfig.read().scenarios
        findings, record = E2eStage(scenarios, ["cuda:0"])._bench_scenario(scenarios[0], tmp_path, {})
        assert not findings
        assert record is not None
        expected = ratio_lower_bound([math.log(6 / ms) for ms in on_ms])
        assert record.latency_ratio_lower_bound == pytest.approx(expected)
        assert record.throughput_ratio_lower_bound == pytest.approx(2)
        payload = record.model_dump()
        assert payload["latency_ratio_lower_bound"] == pytest.approx(expected)
        assert {key for key in payload if key.startswith("latency")} == {"latency_ratio_lower_bound"}
        assert set(payload) == {
            "scenario",
            "n_runs_per_side",
            "throughput_ratio_lower_bound",
            "latency_ratio_lower_bound",
            "n_hit_calls",
            "n_fallback_calls",
        }
        assert len(record.table_row()) == 6
        assert record.table_row()[3] == f"{expected:.4f}"

    @staticmethod
    def _write_run(run_dir: Path, tokens_per_s: float) -> None:
        run_dir.mkdir(parents=True)
        result = {"tokens_per_s": tokens_per_s, "latency_ms": 1 / tokens_per_s}
        (run_dir / "result.json").write_text(json.dumps(result))

    def test_serves_the_bench_measurements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every e2e run's dispatch reads the bench's measurements over the committed
        schemas, at the inherited threshold; the checkout is left as it is."""
        registry = ContractRegistry.default()
        schema = registry.parse_schema(Path(_SCHEMA).read_bytes(), _SCHEMA)
        op_package = registry.get(schema.contract_id).op_package
        bench = BenchReport.from_findings(
            [],
            workload_records=tuple(_workload(w.axes, 2.0, 0.0) for w in schema.workloads),
        )
        earlier = PipelineReport(
            head_sha="",
            base_sha=None,
            measurement_profile={},
            platform=PlatformRecord(**_PLATFORM),
            schema_paths=(_SCHEMA,),
            bench_paths=(_SCHEMA,),
            stages=(AuditReport(status="passed"), bench),
        )
        envs: dict[str, dict[str, str]] = {}

        def execute(run: ScenarioRun) -> None:
            measured_dir = Path(run.env["AKLR_MEASURED_DIR"])
            served, _ = registry.read_schemas(op_package, measured_dir=measured_dir)
            (served_schema,) = [s for p, s in served if p.name == Path(_SCHEMA).name]
            assert {w.speedup for w in served_schema.workloads} == {2.0}
            envs[run.side] = dict(run.env)
            self._write_run(run.run_dir, 1.0)

        monkeypatch.setattr(ScenarioRun, "execute", execute)
        monkeypatch.setattr(_e2e_stage, "read_calls", lambda directory: [])
        monkeypatch.setenv("AKLR_MIN_SPEEDUP", "1.1")
        committed = Path(_SCHEMA).read_bytes()
        E2eStage(E2eConfig.read().scenarios, ["cuda:0"]).run(
            ValidationScope(Path.cwd(), "", None, (), ()), earlier
        )
        assert {e["AKLR_MIN_SPEEDUP"] for e in envs.values()} == {"1.1"}
        assert "AKLR_DISABLE" not in envs["on"]
        assert Path(_SCHEMA).read_bytes() == committed

    @pytest.mark.parametrize(
        ("on_tokens_per_s", "code"),
        [
            pytest.param(2.0, None, id="faster"),
            pytest.param(1.0, FindingCode.E2E_NOT_FASTER, id="no-gain"),
            pytest.param(0.5, FindingCode.E2E_NOT_FASTER, id="slower"),
        ],
    )  # fmt: skip
    def test_gates_performance_with_result_json_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        on_tokens_per_s: float,
        code: FindingCode | None,
    ) -> None:
        """Only timing results are required; performance regressions still fail."""

        def execute(run: ScenarioRun) -> None:
            self._write_run(run.run_dir, 1.0 if run.side == "off" else on_tokens_per_s)

        monkeypatch.setattr(ScenarioRun, "execute", execute)
        monkeypatch.setattr(_e2e_stage, "read_calls", lambda directory: [])
        earlier = Mock()
        earlier.write_schemas.return_value = 0
        report = E2eStage(E2eConfig.read().scenarios[:1], ["cuda:0"]).run(
            ValidationScope(Path.cwd(), "", None, (), ()), earlier
        )
        assert [f.code for f in report.findings] == ([code] if code else [])
        if code is None:
            assert (
                "on 2.0 tokens/s, latency 0.500 ms; off 1.0 tokens/s, latency 1.000 ms, ratio 2.0000"
            ) in capsys.readouterr().err


def test_unkillable_run_raises_instead_of_hanging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A group leader that outlives its SIGKILL (a GPU hang in D state) ends the run
    with RunNotKilledError rather than blocking on its exit."""
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(5)\n")
    scenario = Mock(settings=Mock(timeout_seconds=0.2))
    scenario.argv.return_value = [str(script)]
    monkeypatch.setattr(_e2e_run.os, "killpg", lambda *args: None)
    monkeypatch.setattr(_e2e_run, "_KILL_WAIT_SECONDS", 0.2)
    run = ScenarioRun(scenario, "on", tmp_path / "run", dict(os.environ))
    raised: list[RunNotKilledError] = []

    def execute() -> None:
        try:
            run.execute()
        except RunNotKilledError as e:
            raised.append(e)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert len(raised) == 1
