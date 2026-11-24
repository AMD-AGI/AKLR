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
        (
            f"aklr/plugins/sglang/kimi_k3_tiny_gemm/impls/triton/kimi_k3_tiny_gemm_n8_k8/{_UUID}.json",
            "schema_data",
        ),
        (
            f"aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_model_dim8/{_UUID}.json",
            "schema_data",
        ),
        ("aklr/plugins/sglang/models/qwen2_moe/api.py", "other"),
        ("aklr/plugins/sglang/kimi_k3_tiny_gemm/impls/triton/fp8/fp8_gfx950_m1.json", "other"),
        ("tests/plugins/sglang/test_qwen2_moe_contract.py", "other"),
        ("aklr/core/validation/stages/validation_scope.py", "core"),
        ("stack.json", "core"),
        ("aklr/core/README.md", "other"),
    ],
)
def test_path_kind(path: str, kind: str) -> None:
    assert ValidationScope.path_kind(path) == kind


def test_under_keeps_the_schemas_inside_or_outside_a_directory(tmp_path: Path) -> None:
    sglang, vllm = tmp_path / "aklr/plugins/sglang/a.json", tmp_path / "aklr/plugins/vllm/b.json"
    scope = ValidationScope(tmp_path, "", None, (sglang, vllm), (vllm,))
    assert scope.under("aklr/plugins/sglang") == replace(scope, schema_paths=(sglang,), bench_paths=())
    assert scope.under("aklr/plugins/vllm/").bench_paths == (vllm,)
    assert scope.under("aklr/plugins/sglan").schema_paths == ()
    assert scope.under("aklr/plugins/vllm", inside=False) == replace(
        scope, schema_paths=(sglang,), bench_paths=()
    )
    assert scope.under("aklr/plugins/sglan", inside=False) == scope
    assert scope.under("./aklr/plugins/sglang") == scope.under("aklr/plugins/sglang")
    assert scope.under(".") == scope
    with pytest.raises(ValueError):
        scope.under("..")
