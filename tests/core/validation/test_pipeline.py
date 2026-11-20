# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest

from aklr.core.contract.spec import WorkloadSpec
from aklr.core.validation import (
    AuditStage,
    BenchStage,
    Pipeline,
    PipelineReport,
    PipelineStage,
    ValidationScope,
)
from aklr.core.validation.pipeline import PlatformRecord
from tests.op_case import GEMM_KERNEL_SOURCE, OpCase


class TestPipeline:
    @pytest.mark.parametrize(
        ("stages", "match"),
        [
            pytest.param([], "first stage", id="no_stage"),
            pytest.param([BenchStage(["cuda:0"]), AuditStage()], "first stage", id="bench_first"),
            pytest.param([AuditStage(), AuditStage()], "not in the order", id="repeated"),
        ],
    )
    def test_rejects_stages_out_of_order(self, stages: list[PipelineStage], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline(stages)

    def test_a_failed_audit_skips_the_later_stages(self, tmp_path: Path) -> None:
        path = tmp_path / "aklr/ops/gemm/impls/triton/misplaced/0123456789abcdef.json"
        path.parent.mkdir(parents=True)
        path.write_text(OpCase.of("gemm@1.0").schema().schema_text(), encoding="utf-8")
        scope = ValidationScope(tmp_path, "", None, (path,), (path,))
        report = Pipeline([AuditStage(), BenchStage(["cuda:0"])]).run(scope)
        assert [stage.status for stage in report.stages] == ["failed", "skipped"]
        assert report.exit_code == 1

    def test_the_report_has_no_version(self, tmp_path: Path) -> None:
        scope = ValidationScope(tmp_path, "", None, (), ())
        Pipeline([AuditStage()]).run(scope).write(tmp_path / "out")
        assert "version" not in json.loads((tmp_path / "out/report.json").read_text())


class TestWriteBack:
    _HEAD = "f" * 40

    def _report(
        self,
        tmp_path: Path,
        stored: float | None,
        measured: float,
        microseconds: float | None = None,
    ) -> tuple[str, PipelineReport]:
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
        path = f"aklr/ops/{case.contract.op_type}/{schema.relative_path()}"
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(schema.schema_text(), encoding="utf-8")
        evaluation = WorkloadSpec(
            axes=workload.axes, accuracy="passed", speedup=measured, microseconds=microseconds
        )
        bench = BenchStage.report_type(
            stage_name="bench", status="passed", evaluations={path: {key: evaluation}}
        )
        platform = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
        report = PipelineReport(
            head_sha=self._HEAD,
            base_sha=None,
            platform=platform,
            schema_paths=(path,),
            bench_paths=(path,),
            stages=(bench,),
        )
        return path, report

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
