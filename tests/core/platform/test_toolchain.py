# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.platform import toolchain


@pytest.mark.parametrize(
    ("installed", "unmet"),
    [
        ("3.8.0", None),
        ("3.8.0+amd.rocm7.2.0.git89002410", None),
        ("3.8.0.dev20260901", None),
        ("3.7.0", "triton 3.7.0 is not >=3.8"),
        (None, "triton is not installed, >=3.8 is required"),
    ],
    ids=["release", "local", "dev", "older", "missing"],
)
def test_unmet_requirement(monkeypatch: pytest.MonkeyPatch, installed: str | None, unmet: str | None) -> None:
    """A dev build or a local one counts as its release."""
    monkeypatch.setattr(toolchain, "toolchain_version", lambda name: installed)
    assert toolchain.unmet_requirement({"triton": ">=3.8"}) == unmet
    assert toolchain.unmet_requirement({}) is None


def test_toolchain_version_of_no_module_is_none() -> None:
    assert toolchain.toolchain_version("no_such_toolchain") is None


@pytest.mark.parametrize(
    ("torch_hip", "version"),
    [("7.15.26333", "7.15.26333"), ("7.15.26333-dd7f95766", "7.15.26333"), (None, None)],
    ids=["release", "build_suffix", "not_rocm"],
)
def test_hip_is_the_hip_torch_was_built_with(
    monkeypatch: pytest.MonkeyPatch, torch_hip: str | None, version: str | None
) -> None:
    """No ``hip`` module exists: the hip toolchain is ``torch.version.hip``, which a
    specifier reads once its build suffix is dropped."""
    import torch

    monkeypatch.setattr(torch.version, "hip", torch_hip)
    assert toolchain.toolchain_version("hip") == version
    expected = None if version else "hip is not installed, >=7.15 is required"
    assert toolchain.unmet_requirement({"hip": ">=7.15"}) == expected
