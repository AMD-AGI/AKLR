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
        (f"aklr/plugins/aiter/gemm/impls/triton/gemm_n8_k8/{_UUID}.json", "schema_data"),
        ("aklr/plugins/aiter/gemm/api.py", "other"),
        ("aklr/seam/aiter.py", "other"),
        ("aklr/plugins/aiter/gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/core/test_schema.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("stack.json", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


def test_under_keeps_the_schemas_inside_or_outside_a_directory(tmp_path: Path) -> None:
    ops, plugin = tmp_path / "aklr/plugins/aiter/a.json", tmp_path / "aklr/plugins/b.json"
    scope = ValidationScope(tmp_path, "", None, (ops, plugin), (plugin,))
    assert scope.under("aklr/plugins/aiter") == replace(scope, schema_paths=(ops,), bench_paths=())
    assert scope.under("aklr/plugins/").bench_paths == (plugin,)
    assert scope.under("aklr/op").schema_paths == ()
    assert scope.under("aklr/plugins", inside=False) == replace(scope, schema_paths=(ops,), bench_paths=())
    assert scope.under("aklr/op", inside=False) == scope
    assert scope.under("./aklr/plugins/aiter") == scope.under("aklr/plugins/aiter")
    assert scope.under(".") == scope
    with pytest.raises(ValueError, match="is not in the subpath"):
        scope.under("..")
