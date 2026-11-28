# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which changed paths are schema data, which re-bench every schema, and which schemas
lie under a directory."""

from dataclasses import replace
from pathlib import Path

import pytest

from aklr.core.validation import ValidationScope

_UUID = "0123456789abcdef"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (f"aklr/plugins/aiter/provider/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        (
            f"aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_model_dim8/{_UUID}.json",
            "schema_data",
        ),
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "core"),
        ("aklr/plugins/aiter/provider/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/plugins/sglang/test_qwen2_moe_contract.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("stack.json", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


def test_under_keeps_the_schemas_inside_or_outside_a_directory(tmp_path: Path) -> None:
    aiter, sglang = tmp_path / "aklr/plugins/aiter/a.json", tmp_path / "aklr/plugins/sglang/b.json"
    scope = ValidationScope(tmp_path, "", None, (aiter, sglang), (sglang,))
    assert scope.under("aklr/plugins/aiter") == replace(scope, schema_paths=(aiter,), bench_paths=())
    assert scope.under("aklr/plugins/sglang/").bench_paths == (sglang,)
    assert scope.under("aklr/plugins/ait").schema_paths == ()
    assert scope.under("aklr/plugins/sglang", inside=False) == replace(
        scope, schema_paths=(aiter,), bench_paths=()
    )
    assert scope.under("aklr/plugins/ait", inside=False) == scope
    assert scope.under("./aklr/plugins/aiter") == scope.under("aklr/plugins/aiter")
    assert scope.under(".") == scope
    with pytest.raises(ValueError):
        scope.under("..")


def test_plugin_only_change_selects_full_validation(tmp_path, monkeypatch) -> None:
    import subprocess

    path = tmp_path / f"aklr/plugins/aiter/provider/gemm/impls/triton/gemm_n8_k8/{_UUID}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    (tmp_path / "aklr/plugins/aiter/provider/gemm/contract.py").touch()
    (tmp_path / "aklr/plugins").mkdir(exist_ok=True)
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
