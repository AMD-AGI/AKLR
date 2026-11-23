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
        (f"aklr/ops/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        (
            f"aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_model_dim8/{_UUID}.json",
            "schema_data",
        ),
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "other"),
        ("aklr/ops/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/plugins/sglang/test_qwen2_moe_contract.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("stack.json", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


def test_full_validation_discovers_both_family_layouts_once_in_its_checkout(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    expected = []
    for name, markers in (
        ("new", ["plugin.py"]),
        ("both", ["plugin.py", "contract.py"]),
        ("old", ["contract.py"]),
        ("new/impls/nested", ["plugin.py"]),
    ):
        family = tmp_path / "aklr/plugins/aiter" / name
        schema = family / f"impls/hip/sample/{_UUID}.json"
        schema.parent.mkdir(parents=True)
        schema.touch()
        for marker in markers:
            (family / marker).write_text("raise AssertionError('discovery must not import code')")
        if name != "new/impls/nested":
            expected.append(schema)

    assert ValidationScope.of_working_tree(None, None).schema_paths == tuple(sorted(expected))


def test_under_keeps_the_schemas_inside_or_outside_a_directory(tmp_path: Path) -> None:
    ops, plugin = tmp_path / "aklr/ops/a.json", tmp_path / "aklr/plugins/b.json"
    scope = ValidationScope(tmp_path, "", None, (ops, plugin), (plugin,))
    assert scope.under("aklr/ops") == replace(scope, schema_paths=(ops,), bench_paths=())
    assert scope.under("aklr/plugins/").bench_paths == (plugin,)
    assert scope.under("aklr/op").schema_paths == ()
    assert scope.under("aklr/plugins", inside=False) == replace(scope, schema_paths=(ops,), bench_paths=())
    assert scope.under("aklr/op", inside=False) == scope
    assert scope.under("./aklr/ops") == scope.under("aklr/ops")
    assert scope.under(".") == scope
    with pytest.raises(ValueError):
        scope.under("..")


def test_a_moved_measured_schema_retains_its_base_and_has_no_new_benchmark(tmp_path, monkeypatch):
    import subprocess

    root = tmp_path
    relative = Path("rms_norm/impls/hip/add_rms_norm_hidden_size6144/1f0b6c68db064ca4.json")
    original = root / "aklr/ops" / relative
    original.parent.mkdir(parents=True)
    original.write_bytes((Path("aklr/plugins/aiter") / relative).read_bytes())

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "before")
    current = root / "aklr/plugins/aiter" / relative
    current.parent.mkdir(parents=True)
    original.rename(current)
    monkeypatch.chdir(root)
    scope = ValidationScope.of_working_tree("HEAD", None)
    assert scope.schema_paths == (current,)
    assert scope.bench_paths == ()
    assert not scope.is_first_release
    assert original.relative_to(root).as_posix() in scope.schema_paths_at_base(
        "aklr/plugins/aiter/rms_norm/impls"
    )

    from aklr.core.validation.stages import AuditStage

    audit = AuditStage().run(scope, ())
    assert not audit.findings
    original.parent.mkdir(exist_ok=True, parents=True)
    original.write_bytes(current.read_bytes())
    assert any(f.code.value == "perf_edited" for f in AuditStage().run(scope, ()).findings)


def test_multiple_scope_roots_select_a_union_then_remove_exclusions(tmp_path):
    a, b, c = [tmp_path / part / "file.json" for part in ("a", "b", "c")]
    scope = ValidationScope(tmp_path, "", None, (a, b, c), (b, c))
    selected = scope.under("a", "b").under("c", "a", inside=False)
    assert selected.schema_paths == selected.bench_paths == (b,)
