# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from aklr.core.jit import flydsl_cache
from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX
from aklr.core.runtime.builder.flydsl_fastcall import install

flydsl = pytest.importorskip("flydsl")

# out accumulates, so a call that launched twice shows; OFFSET is a global FlyDSL checks on each of its calls
_LAUNCHER_SOURCE = """\
import flydsl.compiler as flyc
import flydsl.expr as fx

OFFSET = 0


@flyc.kernel
def scale_kernel(x: fx.Tensor, out: fx.Tensor, factor: fx.Int32):
    i = fx.thread_idx.x
    out[i] = out[i] + x[i] * factor + OFFSET


@flyc.jit
def launch(x: fx.Tensor, out: fx.Tensor, factor: fx.Int32, n: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    scale_kernel(x, out, factor).launch(grid=(1, 1, 1), block=(n, 1, 1), stream=stream)
"""
_DRIFT = "changed since first compile"


def _module(tmp_path: Path, name: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(_LAUNCHER_SOURCE)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_install_leaves_modern_launchers_to_the_existing_cache(monkeypatch):
    """FlyDSL 0.5 and later use AKLR's regular launch cache, including its specialization keys."""
    from flydsl.compiler.jit_function import JitFunction

    monkeypatch.setattr(flydsl, "__version__", "0.5.0")
    original = JitFunction.__call__
    monkeypatch.setattr(JitFunction, "__call__", original)
    install(KERNEL_PACKAGE_PREFIX)
    assert JitFunction.__call__ is original


@pytest.fixture
def modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The launcher's module in AKLR's kernel packages and in another, with the fast path installed
    until the test ends."""
    from flydsl.compiler.jit_function import JitFunction

    if flydsl_cache.is_needed():
        pytest.skip("FlyDSL 0.5 and later use the regular AKLR launch cache")
    monkeypatch.setattr(JitFunction, "__call__", JitFunction.__call__)
    install(KERNEL_PACKAGE_PREFIX)
    names = (f"{KERNEL_PACKAGE_PREFIX}fastcall_case", "fastcall_other_case")
    try:
        yield tuple(_module(tmp_path, name) for name in names)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def _scaled(launch, n: int, factor: int, *args, **kwargs) -> None:
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(x)
    launch(x, out, factor, n, *args, **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * factor, rtol=0, atol=0)


def _streams():
    import flydsl.expr as fx

    return [
        ((), {}),
        ((fx.Stream(None),), {}),
        ((), {"stream": fx.Stream(None)}),
        ((torch.cuda.current_stream(),), {}),
        ((), {"stream": torch.cuda.current_stream()}),
    ]


@pytest.mark.gpu
def test_each_call_launches_once_exactly(modules) -> None:
    """Repeated calls and calls of another shape, scalar or stream, positional, keyword or
    default, each launch once with the result FlyDSL's path gives."""
    launch = modules[0].launch
    for args, kwargs in _streams():
        for n, factor in ((64, 3), (64, 3), (64, 5), (32, 3)):
            _scaled(launch, n, factor, *args, **kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("by_keyword", [False, True], ids=["positional", "keyword"])
def test_hit_launches_on_the_given_stream(modules, by_keyword: bool) -> None:
    """A repeated launch on a side stream runs there: it waits on that stream's earlier work."""
    launch = modules[0].launch
    side = torch.cuda.Stream()
    x = torch.arange(64, dtype=torch.int32, device="cuda")
    for _ in range(2):
        out = torch.zeros_like(x)
        with torch.cuda.stream(side):
            torch.cuda._sleep(50_000_000)
            src = torch.arange(64, dtype=torch.int32, device="cuda")
        if by_keyword:
            launch(src, out, 3, 64, stream=side)
        else:
            launch(src, out, 3, 64, side)
        side.synchronize()
        torch.testing.assert_close(out, x * 3, rtol=0, atol=0)


@pytest.mark.gpu
def test_hit_skips_flydsl_checks_until_a_new_signature(modules) -> None:
    """A repeated signature launches what was compiled, past a changed global; a new one takes
    FlyDSL's path, which raises on it."""
    module = modules[0]
    _scaled(module.launch, 64, 3)
    module.OFFSET = 1
    _scaled(module.launch, 64, 3)
    with pytest.raises(RuntimeError, match=_DRIFT):
        _scaled(module.launch, 32, 3)


@pytest.mark.gpu
def test_other_packages_take_flydsl_path(modules) -> None:
    """A launcher outside AKLR's kernel packages runs FlyDSL's checks on every call."""
    module = modules[1]
    _scaled(module.launch, 64, 3)
    module.OFFSET = 1
    with pytest.raises(RuntimeError, match=_DRIFT):
        _scaled(module.launch, 64, 3)


@pytest.mark.gpu
def test_unknown_argument_takes_flydsl_path(modules) -> None:
    """A call with an argument the fast path cannot key runs FlyDSL's checks every time."""
    import flydsl.expr as fx

    module = modules[0]
    x = torch.arange(64, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(x)
    module.launch(x, out, fx.Int32(3), 64)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 3, rtol=0, atol=0)
    module.OFFSET = 1
    with pytest.raises(RuntimeError, match=_DRIFT):
        module.launch(x, out, fx.Int32(3), 64)


@pytest.mark.gpu
def test_kernel_compiling_its_launcher_launches_once(modules) -> None:
    """A kernel that ``flyc.compile``s its launcher itself, as aiter's do, launches once on that
    call and through the compiled function after."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    launch = modules[0].launch
    x = torch.arange(64, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(x)
    compiled = flyc.compile(launch, x, out, 3, 64, fx.Stream(None))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 3, rtol=0, atol=0)
    compiled(x, out, 3, 64, fx.Stream(None))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 6, rtol=0, atol=0)
