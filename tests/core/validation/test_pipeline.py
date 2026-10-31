# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import Schema, load_json_file, save_json_file
from aklr.core.validation import (
    AuditStage,
    BenchStage,
    Pipeline,
    PipelineReport,
    PipelineStage,
    ValidationScope,
)
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
        save_json_file(OpCase.of("gemm_bf16_nt").schema(), path)
        scope = ValidationScope(tmp_path, "", None, (path,), (path,))
        report = Pipeline([AuditStage(), BenchStage(["cuda:0"])]).run(scope)
        assert [stage.status for stage in report.stages] == ["failed", "skipped"]
        assert report.exit_code == 1


@pytest.mark.gpu
@pytest.mark.gfx950
class TestBench:
    @staticmethod
    def _run(root: Path, source: str) -> PipelineReport:
        schema = OpCase.of("gemm_bf16_nt").served('"""gemm."""\n' + source, evaluations=[])
        path = root / "aklr/ops/gemm/impls/triton" / schema.name / f"{schema.uuid}.json"
        save_json_file(schema, path)
        scope = ValidationScope(root, "head", None, (path,), (path,))
        return Pipeline([AuditStage(), BenchStage(["cuda:0"])]).run(scope)

    def test_writes_back_a_correct_kernel_as_passed(self, tmp_path: Path) -> None:
        report = self._run(tmp_path, GEMM_KERNEL)
        assert report.write_schemas(tmp_path, tmp_path / "out") == 1
        (path,) = (tmp_path / "out").rglob("*.json")
        assert [
            w.evaluation(DeviceTarget.current()).accuracy for w in load_json_file(Schema, path).workloads
        ] == ["passed"]

    def test_a_wrong_kernel_is_a_numeric_mismatch(self, tmp_path: Path) -> None:
        report = self._run(tmp_path, GEMM_KERNEL.replace("a @ b.mT", "a @ b.mT + 1"))
        assert "numeric_mismatch" in {f.code for f in report.stages[1].demotions}
