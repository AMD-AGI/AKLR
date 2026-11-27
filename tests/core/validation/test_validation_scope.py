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
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "other"),
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
