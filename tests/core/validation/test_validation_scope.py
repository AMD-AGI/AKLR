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
