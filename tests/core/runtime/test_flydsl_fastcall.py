# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX
from aklr.core.runtime.builder._flydsl_fastcall import install

pytest.importorskip("flydsl")

_LAUNCHER_SOURCE = """\
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def scale_kernel(x: fx.Tensor, out: fx.Tensor, factor: fx.Int32):
    i = fx.thread_idx.x
    out[i] = x[i] * factor


@flyc.jit
def launch(x: fx.Tensor, out: fx.Tensor, factor: fx.Int32, n: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    scale_kernel(x, out, factor).launch(grid=(1, 1, 1), block=(n, 1, 1), stream=stream)
"""


def _module(tmp_path: Path, name: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(_LAUNCHER_SOURCE)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launchers(tmp_path: Path):
    """The launcher in a module of AKLR's kernel packages and in another, with the fast path installed."""
    install(KERNEL_PACKAGE_PREFIX)
    names = (f"{KERNEL_PACKAGE_PREFIX}fastcall_case", "fastcall_other_case")
    try:
        yield tuple(_module(tmp_path, name).launch for name in names)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def _scaled(launch, n: int, factor: int, *, arg=None, **kwargs) -> None:
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(x)
    launch(x, out, factor if arg is None else arg, n, **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * factor, rtol=0, atol=0)


@pytest.mark.gpu
class TestFlyDSLFastcall:
    """AKLR's launchers keep a compiled function per argument signature; the rest take FlyDSL's path."""

    def test_one_entry_per_signature(self, launchers) -> None:
        """Repeated calls reuse an entry; a new shape, scalar or keyword makes another, each exact."""
        import flydsl.expr as fx

        launch, _ = launchers
        _scaled(launch, 64, 3)
        _scaled(launch, 64, 3)
        assert len(launch._aklr_compiled) == 1
        _scaled(launch, 64, 5)
        _scaled(launch, 32, 3)
        _scaled(launch, 64, 3, stream=fx.Stream(None))
        _scaled(launch, 64, 7, stream=fx.Stream(None))
        _scaled(launch, 64, 3, stream=torch.cuda.current_stream())
        _scaled(launch, 64, 3, stream=torch.cuda.current_stream())
        assert len(launch._aklr_compiled) == 6

    def test_hit_launches_on_the_given_stream(self, launchers) -> None:
        """A cached launch on a side stream runs there: it waits on that stream's earlier work."""
        launch, _ = launchers
        _scaled(launch, 64, 3, stream=torch.cuda.current_stream())
        side = torch.cuda.Stream()
        x = torch.zeros(64, dtype=torch.int32, device="cuda")
        out = torch.zeros_like(x)
        with torch.cuda.stream(side):
            torch.cuda._sleep(50_000_000)
            x.copy_(torch.arange(64, dtype=torch.int32, device="cuda"))
        launch(x, out, 3, 64, stream=side)
        side.synchronize()
        torch.testing.assert_close(
            out, torch.arange(64, dtype=torch.int32, device="cuda") * 3, rtol=0, atol=0
        )

    def test_other_packages_take_flydsl_path(self, launchers) -> None:
        """A launcher outside AKLR's kernel packages is marked once and never cached."""
        _, other = launchers
        _scaled(other, 64, 3)
        _scaled(other, 64, 3)
        assert other._aklr_compiled is False

    def test_unknown_argument_takes_flydsl_path(self, launchers) -> None:
        """An argument the signature cannot key launches through FlyDSL, adding no entry."""
        import flydsl.expr as fx

        launch, _ = launchers
        _scaled(launch, 64, 3, arg=fx.Int32(3))
        assert launch._aklr_compiled == {}
