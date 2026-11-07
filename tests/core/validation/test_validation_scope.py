# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which changed paths are schema data, and which are aklr/core code."""

import pytest

from aklr.core.validation import ValidationScope

_UUID = "0123456789abcdef"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (f"aklr/ops/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        (
            f"aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_model_dim8/{_UUID}.json",
            "schema_data",
        ),
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "core"),
        ("aklr/ops/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/plugins/sglang/test_qwen2_moe_contract.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


def test_plugin_only_change_selects_full_validation(tmp_path, monkeypatch) -> None:
    import subprocess

    path = tmp_path / f"aklr/ops/gemm/impls/triton/gemm_n8_k8/{_UUID}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    (tmp_path / "aklr/plugins").mkdir()
    monkeypatch.chdir(tmp_path)

    def git(argv, **kwargs):
        args = argv[5:]
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            result = str(tmp_path).encode()
        elif args[0] == "rev-parse":
            result = b"head"
        elif args[0] == "merge-base":
            result = b"base"
        elif args[0] == "diff":
            result = b"aklr/plugins/sglang/routes.py\0"
        elif args[0] == "ls-files":
            result = b""
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, result, b"")

    monkeypatch.setattr(subprocess, "run", git)
    scope = ValidationScope.of_working_tree("main", None)
    assert scope.schema_paths == scope.bench_paths == (path,)
