# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which changed paths are schema data, and which are aklr/core code."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import Evaluation, Workload, save_json_file
from aklr.core.validation import ValidationScope
from tests.op_sample import GEMM_KERNEL, GEMM_SAMPLE

_UUID = "0123456789abcdef"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (f"aklr/ops/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        ("aklr/ops/gemm/api.py", "other"),
        ("aklr/seam/aiter.py", "other"),
        ("aklr/ops/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/core/test_schema.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


@pytest.mark.parametrize(
    ("source", "workload", "is_benched"),
    [
        pytest.param(GEMM_KERNEL + "# edited\n", Workload(axes={"M": 4}), True, id="sources_edited"),
        pytest.param(GEMM_KERNEL, Workload(axes={"M": 8}), True, id="workload_axes_edited"),
        pytest.param(
            GEMM_KERNEL,
            Workload(axes={"M": 4}, evaluations=[Evaluation(hardware="gfx950/MI355X", accuracy="passed")]),
            False,
            id="measured",
        ),
    ],
)
def test_the_bench_skips_a_path_at_base_when_only_evaluations_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    workload: Workload,
    is_benched: bool,
) -> None:
    monkeypatch.setattr(DeviceTarget, "current", lambda: DeviceTarget.from_names("gfx950", "MI355X"))
    committed = GEMM_SAMPLE.served(GEMM_KERNEL, evaluations=[])
    path = tmp_path.resolve() / f"aklr/ops/gemm/impls/triton/{committed.name}/{committed.uuid}.json"
    save_json_file(committed, path)
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=tmp_path, check=True)
    save_json_file(replace(GEMM_SAMPLE.served(source, evaluations=[]), workloads=[workload]), path)
    monkeypatch.chdir(tmp_path)
    scope = ValidationScope.of_working_tree("HEAD", None)
    assert (scope.schema_paths, scope.bench_paths) == ((path,), (path,) if is_benched else ())
