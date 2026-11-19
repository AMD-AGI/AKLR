# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which changed paths are schema data, which re-bench every schema, and which schemas
lie under a directory."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from aklr.core.validation import ValidationScope

_UUID = "0123456789abcdef"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (f"aklr/plugins/aiter/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        (
            f"aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_model_dim8/{_UUID}.json",
            "schema_data",
        ),
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "other"),
        ("aklr/plugins/aiter/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/plugins/sglang/test_qwen2_moe_contract.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("stack.json", "core"),
        ("aklr/core/README.md", "other"),
        ("traffic/schemas.json", "traffic"),
        ("traffic/a.traffic.jsonl.xz", "traffic"),
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


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=x@x", "-c", "user.name=x", *args],
        check=True,
        capture_output=True,
    )


def _listing(root: Path, samples: int, schemas: dict[str, list[str]]) -> None:
    listed = {name: {"recordings": recordings} for name, recordings in schemas.items()}
    (root / "traffic/schemas.json").write_text(json.dumps({"samples": samples, "schemas": listed}))


@pytest.mark.parametrize(
    ("change", "benched"),
    [
        (lambda root: (root / "traffic/r.traffic.jsonl.xz").write_bytes(b"new"), ["a"]),
        (lambda root: _listing(root, 3, {"a": ["r"], "b": ["s"]}), ["b"]),
        (lambda root: _listing(root, 5, {"a": ["r"]}), ["a"]),
        (lambda root: _listing(root, 3, {}), ["a"]),
        (lambda root: (root / "traffic/README.md").write_text("new"), []),
    ],
    ids=["a recording a schema lists", "a schema listed", "samples", "a schema unlisted", "the readme"],
)
def test_a_traffic_change_benches_the_schemas_measured_on_it(tmp_path, monkeypatch, change, benched):
    for name in ("a", "b"):
        schema = tmp_path / f"aklr/plugins/aiter/op/impls/triton/{name}/{_UUID}.json"
        schema.parent.mkdir(parents=True)
        schema.write_text("{}")
    (tmp_path / "aklr/plugins/aiter/op/contract.py").write_text("")
    (tmp_path / "traffic").mkdir()
    (tmp_path / "traffic/r.traffic.jsonl.xz").write_bytes(b"old")
    (tmp_path / "traffic/README.md").write_text("old")
    _listing(tmp_path, 3, {"a": ["r"]})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "branch", "base")
    change(tmp_path)
    monkeypatch.chdir(tmp_path)
    scope = ValidationScope.of_working_tree("base", None)
    assert [p.parent.name for p in scope.bench_paths] == benched
    assert [p.parent.name for p in scope.schema_paths] == benched
