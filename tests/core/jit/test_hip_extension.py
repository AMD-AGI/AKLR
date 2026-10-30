# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest
import torch

from aklr.core.jit import hip_extension

_FILL_SOURCE = """\
// fill(out, value): an extension with no device code.
#include <ATen/ATen.h>
#include <torch/library.h>

namespace {
void fill(at::Tensor &out, double value) { out.fill_(value); }
} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("fill(Tensor(a!) out, float value) -> ()", &fill);
}
"""


def _load(name: str, root: Path) -> Path:
    return hip_extension.load_library(name, {"fill.hip": _FILL_SOURCE}, root, "gfx950")


@pytest.mark.gpu
class TestLoadLibrary:
    """``load_library`` builds an extension once, then loads it."""

    def test_builds_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        library = _load("aklr_test_fill_once", tmp_path)
        out = torch.zeros(4)
        torch.ops.aklr_test_fill_once.fill(out, 2.0)
        assert out.eq(2.0).all()
        from torch.utils import cpp_extension

        monkeypatch.setattr(cpp_extension, "load", lambda **_: pytest.fail("rebuilt"))
        assert _load("aklr_test_fill_once", tmp_path) == library

    def test_clears_a_stale_torch_lock(self, tmp_path: Path) -> None:
        """A build that died leaves torch's lock file, which would block every later build."""
        library = _load("aklr_test_fill_lock", tmp_path)
        (library.parent / "built").unlink()
        (library.parent / "lock").touch()
        assert _load("aklr_test_fill_lock", tmp_path) == library
        assert not (library.parent / "lock").exists()
