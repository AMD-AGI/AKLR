# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest

from aklr.core.contract.spec import WorkloadSpec
from aklr.core.validation import (
    AuditReport,
    AuditValidator,
    BenchReport,
    BenchValidator,
    ValidationReport,
    ValidationScope,
    Validator,
    ValidatorReport,
    validate,
)
from aklr.core.validation.report import PlatformRecord
from tests.op_case import GEMM_KERNEL_SOURCE, OpCase


class _Scripted(Validator):
    def __init__(self, name: str, status: str) -> None:
        self.name = name
        self.status = status
        self.ran = False

    def run(self, scope: ValidationScope, platform: PlatformRecord) -> ValidatorReport:
        self.ran = True
        return self.report_type(status=self.status)


def _scope_of_one_schema(root: Path) -> ValidationScope:
    path = root / "aklr/plugins/aiter/provider/gemm/impls/triton/kernel/0123456789abcdef.json"
    return ValidationScope(root, "", None, (path,), (path,))


def _report(*validators: ValidatorReport) -> ValidationReport:
    platform = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
    return ValidationReport(
        head_sha="f" * 40,
        base_sha=None,
        platform=platform,
        schema_paths=(),
        bench_paths=(),
        validators=validators,
    )


class TestValidate:
    def test_a_validator_runs_whatever_another_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failing, passing = _Scripted("audit", "failed"), _Scripted("bench", "passed")
        report = validate(_scope_of_one_schema(tmp_path), [failing, passing])
        assert failing.ran and passing.ran
        assert [(v.name, v.status) for v in report.validators] == [("audit", "failed"), ("bench", "passed")]
        assert report.conclusion == "failed (audit)"
        assert report.exit_code == 1
        progress = capsys.readouterr().err
        assert "audit: started\n" in progress
        assert "bench: started\n" in progress

    def test_the_bench_is_entered_without_an_audit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = validate(_scope_of_one_schema(tmp_path), [BenchValidator(["cuda:0"])])
        assert [(v.name, v.status) for v in report.validators] == [("bench", "aborted")]
        assert report.conclusion == "aborted (bench)"
        assert report.exit_code == 3
        assert "bench: started\n" in capsys.readouterr().err

    @pytest.mark.parametrize("schema_paths", [0, 1], ids=["no_schema", "a_schema"])
    def test_a_run_of_no_validator_is_an_error(self, tmp_path: Path, schema_paths: int) -> None:
        paths = _scope_of_one_schema(tmp_path).schema_paths[:schema_paths]
        scope = ValidationScope(tmp_path, "", None, paths, paths)
        with pytest.raises(ValueError, match="no validator to run"):
            validate(scope, [])

    @pytest.mark.parametrize(
        ("statuses", "conclusion", "exit_code"),
        [
            pytest.param(("passed", "passed"), "passed", 0, id="all_passed"),
            pytest.param(("passed", "skipped"), "passed", 0, id="skipped_is_not_a_failure"),
            pytest.param(("passed", "failed"), "failed (bench)", 1, id="one_failed"),
            pytest.param(("passed", "aborted"), "aborted (bench)", 3, id="one_aborted"),
            pytest.param(
                ("failed", "aborted"), "failed (audit), aborted (bench)", 1, id="failed_over_aborted"
            ),
            pytest.param(("aborted", "failed"), "aborted (audit), failed (bench)", 1, id="failed_wherever"),
            pytest.param(("aborted", "aborted"), "aborted (audit), aborted (bench)", 3, id="both_aborted"),
        ],
    )
    def test_the_conclusion_lists_every_validator_that_did_not_pass(
        self, statuses: tuple[str, str], conclusion: str, exit_code: int
    ) -> None:
        report = _report(
            AuditReport(status=statuses[0]),
            BenchReport(status=statuses[1]),
        )
        assert report.conclusion == conclusion
        assert report.exit_code == exit_code

    def test_the_summary_table_has_a_row_per_validator(self) -> None:
        report = _report(
            BenchReport(status="failed", elapsed_secs=2.0),
            AuditReport(status="skipped", detail="no schema to validate"),
        )
        table = [line for line in report.markdown_text().splitlines() if line.startswith("|")]
        assert [[cell.strip() for cell in line.strip("|").split("|")] for line in table[:1] + table[2:]] == [
            ["validator", "status", "findings", "elapsed seconds"],
            ["bench", "failed", "0", "2.0"],
            ["audit", "skipped: no schema to validate", "-", "-"],
        ]

    def test_the_report_names_its_validators(self, tmp_path: Path) -> None:
        _report(BenchReport(status="passed")).write(tmp_path)
        written = json.loads((tmp_path / "report.json").read_text())
        assert [(v["name"], v["status"]) for v in written["validators"]] == [("bench", "passed")]

    def test_the_report_has_no_version(self, tmp_path: Path) -> None:
        scope = ValidationScope(tmp_path, "", None, (), ())
        validate(scope, [AuditValidator()]).write(tmp_path / "out")
        assert "version" not in json.loads((tmp_path / "out/report.json").read_text())


class TestWriteBack:
    _HEAD = "f" * 40

    def _report(
        self,
        tmp_path: Path,
        stored: float | None,
        measured: float,
        microseconds: float | None = None,
    ) -> tuple[str, ValidationReport]:
        """A schema of one workload stored passed at the speedup stored (None: unmeasured), and the report of a
        bench that measured it passed at measured."""
        case = OpCase.of("gemm@1.0")
        schema = case.schema(GEMM_KERNEL_SOURCE)
        (workload,) = schema.workloads
        key = ",".join(f"{a}={v}" for a, v in workload.axes.items())
        if stored is not None:
            schema = schema.with_measurements(
                {key: WorkloadSpec(axes=workload.axes, accuracy="passed", speedup=stored)}
            )
        path = f"aklr/plugins/aiter/provider/{case.contract.op_type}/{schema.relative_path()}"
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(schema.schema_text(), encoding="utf-8")
        evaluation = WorkloadSpec(
            axes=workload.axes, accuracy="passed", speedup=measured, microseconds=microseconds
        )
        bench = BenchReport(status="passed", evaluations={path: {key: evaluation}})
        platform = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
        report = ValidationReport(
            head_sha=self._HEAD,
            base_sha=None,
            platform=platform,
            schema_paths=(path,),
            bench_paths=(path,),
            validators=(bench,),
        )
        return path, report

    def test_a_failed_audit_does_not_hold_the_measurements_back(self, tmp_path: Path) -> None:
        path, report = self._report(tmp_path, 1.5, 1.6)
        audit = AuditReport(status="failed")
        report = ValidationReport(
            head_sha=report.head_sha,
            base_sha=None,
            platform=report.platform,
            schema_paths=report.schema_paths,
            bench_paths=report.bench_paths,
            validators=(audit, *report.validators),
        )
        assert list(report.measured_schemas(tmp_path)) == [path]

    def test_a_speedup_within_two_percent_rewrites_nothing(self, tmp_path: Path) -> None:
        _, report = self._report(tmp_path, 1.5, 1.51)
        assert report.measured_schemas(tmp_path) == {}

    def test_a_moved_speedup_rewrites_the_file(self, tmp_path: Path) -> None:
        path, report = self._report(tmp_path, 1.5, 1.6)
        assert report.measured_schemas(tmp_path)[path].workloads[0].speedup == 1.6

    def test_a_time_that_comes_rewrites_the_file_with_it(self, tmp_path: Path) -> None:
        # a time that drifts rides with its speedup; one that comes or goes is a change of its own
        path, report = self._report(tmp_path, 1.5, 1.51, microseconds=12.3)
        assert report.measured_schemas(tmp_path)[path].workloads[0].microseconds == 12.3

    def test_an_unmeasured_workload_is_written(self, tmp_path: Path) -> None:
        path, report = self._report(tmp_path, None, 1.0)
        assert path in report.measured_schemas(tmp_path)

    def test_write_schemas_writes_the_rewritten_file(self, tmp_path: Path) -> None:
        path, report = self._report(tmp_path, 1.5, 1.6)
        assert report.write_schemas(tmp_path, tmp_path / "out") == 1
        assert json.loads((tmp_path / "out" / path).read_text())["workloads"][0]["speedup"] == 1.6
