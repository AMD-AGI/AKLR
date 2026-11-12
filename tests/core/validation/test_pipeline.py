# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest

from aklr.core.schema import (
    Build,
    Evaluation,
    Kernel,
    SourceFile,
    Speedup,
    Workload,
    save_json_file,
)
from aklr.core.validation import (
    AuditStage,
    BenchStage,
    Pipeline,
    PipelineReport,
    PipelineStage,
    ValidationScope,
)
from aklr.core.validation.pipeline import PlatformRecord
from tests.op_case import GEMM_KERNEL, OpCase


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
        save_json_file(OpCase.from_id("gemm_bf16_nt").schema(), path)
        scope = ValidationScope(tmp_path, "", None, (path,), (path,))
        report = Pipeline([AuditStage(), BenchStage(["cuda:0"])]).run(scope)
        assert [stage.status for stage in report.stages] == ["failed", "skipped"]
        assert report.exit_code == 1

    def test_the_report_is_of_version_3(self, tmp_path: Path) -> None:
        scope = ValidationScope(tmp_path, "", None, (), ())
        Pipeline([AuditStage()]).run(scope).write(tmp_path / "out")
        assert json.loads((tmp_path / "out/report.json").read_text())["version"] == 3


class TestWriteBack:
    _PATH = "aklr/ops/gemm/impls/triton/gemm_bf16_nt_n32_k64/0123456789abcdef.json"
    _HEAD = "f" * 40

    def _report(self, tmp_path: Path, stored: list[Evaluation], measured: float) -> PipelineReport:
        kernel = Kernel(
            name="gemm",
            author="tests",
            build=Build(language="triton", target_hardware=["gfx950"], entry_point="kernel.py::run"),
            sources=[SourceFile(path="kernel.py", content=GEMM_KERNEL)],
        )
        schema = OpCase.from_id("gemm_bf16_nt").schema(kernel, [Workload(axes={"M": 1}, evaluations=stored)])
        save_json_file(schema, tmp_path / self._PATH)
        evaluation = Evaluation(
            hardware="gfx950/MI355X", accuracy="passed", speedup={"aiter": Speedup(self._HEAD, measured)}
        )
        bench = BenchStage.report_type(
            stage_name="bench", status="passed", evaluations={self._PATH: {"M=1": evaluation}}
        )
        platform = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
        return PipelineReport(
            head_sha=self._HEAD,
            base_sha=None,
            platform=platform,
            schema_paths=(self._PATH,),
            bench_paths=(self._PATH,),
            stages=(bench,),
        )

    @staticmethod
    def _stored(value: float) -> list[Evaluation]:
        return [
            Evaluation(hardware="gfx942/MI300X", accuracy="failed"),
            Evaluation(
                hardware="gfx950/MI355X", accuracy="passed", speedup={"aiter": Speedup("e" * 40, value)}
            ),
        ]

    def test_a_speedup_within_two_percent_rewrites_nothing(self, tmp_path: Path) -> None:
        assert self._report(tmp_path, self._stored(1.5), 1.51).measured_schemas(tmp_path) == {}

    def test_a_moved_speedup_rewrites_this_hardware_only(self, tmp_path: Path) -> None:
        rewritten = self._report(tmp_path, self._stored(1.5), 1.6).measured_schemas(tmp_path)
        evaluations = rewritten[self._PATH].workloads[0].evaluations
        assert evaluations[0] == Evaluation(hardware="gfx942/MI300X", accuracy="failed")
        assert evaluations[1].speedup["aiter"] == Speedup(self._HEAD, 1.6)

    def test_an_unmeasured_workload_is_written(self, tmp_path: Path) -> None:
        assert self._PATH in self._report(tmp_path, [], 1.0).measured_schemas(tmp_path)
