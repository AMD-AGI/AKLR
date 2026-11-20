# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launch cache aklr.core.jit.flydsl_cache puts back for AKLR's FlyDSL kernels on FlyDSL 0.5 (CPU): which calls
it keeps, under which key, and what it leaves to FlyDSL. A real JitFunction, imported from a kernel package as
FlyDSLBuilder names them; FlyDSL's compile and its own call path are stand-ins."""

import importlib
import sys

import pytest
import torch

flydsl = pytest.importorskip("flydsl")
flydsl_jit = pytest.importorskip("flydsl.compiler.jit_function")

from aklr.core.jit import flydsl_cache  # noqa: E402

pytestmark = pytest.mark.skipif(not flydsl_cache.is_needed(), reason="FlyDSL before 0.5 keeps its own cache")

KERNEL = """
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def launch(x: fx.Tensor, n: fx.Int32, block: fx.Constexpr[int], scale=1):
    pass
"""


_PACKAGES = ("_aklr_flydsl_0123456789abcdef", "aiter_like")


@pytest.fixture(autouse=True)
def _kernel_modules_leave_with_the_test():
    """The kernel packages the tests import: a later test (validation's trial) checks none is imported."""
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in _PACKAGES]:
        del sys.modules[name]


def _jit_function(tmp_path, monkeypatch, package):
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / package / "kernel.py").write_text(KERNEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(f"{package}.kernel").launch


class _Compiled:
    """A CompiledFunction: records the positional values of each launch."""

    def __init__(self):
        self.launches = []

    def __call__(self, *values):
        self.launches.append(values)


@pytest.fixture
def flydsl_paths(monkeypatch):
    """flyc.compile and FlyDSL's own call path as stand-ins; every tensor counts as on a device."""
    paths = type("Paths", (), {})()
    paths.compiled, paths.own = [], []

    def compile_impl(jit_function, *args, **kwargs):
        compiled = _Compiled()
        paths.compiled.append((compiled, args, kwargs))
        return compiled

    monkeypatch.setattr(flydsl_jit, "_compile_impl", compile_impl)
    monkeypatch.setattr(flydsl_cache, "_original_call", lambda jf, *a, **k: paths.own.append((jf, a, k)))
    monkeypatch.setattr(flydsl_cache, "_would_launch", lambda values: True)
    return paths


@pytest.fixture
def launch(tmp_path, monkeypatch):
    return _jit_function(tmp_path, monkeypatch, _PACKAGES[0])


def _call(jit_function, *args, **kwargs):
    return flydsl_cache._call(jit_function, args, kwargs)


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def test_a_specialization_compiles_once_and_then_launches_its_compiled_function(launch, flydsl_paths):
    x = bf16(4, 8)
    _call(launch, x, 5, 64)
    _call(launch, x, 6, block=64)
    ((compiled, args, _),) = flydsl_paths.compiled
    assert args == (x, 5, 64)
    # every parameter by position, defaults applied
    assert compiled.launches == [(x, 6, 64, 1)]
    assert flydsl_paths.own == []


def test_runtime_values_and_a_dynamic_layouts_shape_share_the_specialization(launch, flydsl_paths):
    _call(launch, bf16(4, 8), 5, 64)
    _call(launch, bf16(300, 8), 7, 64, 3)
    assert len(flydsl_paths.compiled) == 1


@pytest.mark.parametrize(
    "other",
    [(bf16(4, 8), 5, 128), (torch.zeros(4, 8), 5, 64), (bf16(8, 4).t(), 5, 64)],
    ids=["constexpr", "dtype", "unit_stride_axis"],
)
def test_what_flydsl_specializes_on_compiles_again(launch, flydsl_paths, other):
    _call(launch, bf16(4, 8), 5, 64)
    _call(launch, *other)
    assert len(flydsl_paths.compiled) == 2


def test_compile_hints_and_the_environment_are_in_the_key(launch, flydsl_paths, monkeypatch):
    _call(launch, bf16(4, 8), 5, 64)
    monkeypatch.setattr(type(launch), "_effective_compile_hints", lambda self: {"fast_fp_math": True})
    _call(launch, bf16(4, 8), 5, 64)
    env = flydsl_jit._cache_invalidating_env_values(flydsl_jit.get_backend().target)
    if env:
        monkeypatch.setenv(env[0][0], "changed")
        _call(launch, bf16(4, 8), 5, 64)
    assert len(flydsl_paths.compiled) == (3 if env else 2)


def test_a_function_no_aklr_kernel_defines_keeps_flydsls_path(tmp_path, monkeypatch, flydsl_paths):
    theirs = _jit_function(tmp_path, monkeypatch, _PACKAGES[1])
    x = bf16(4, 8)
    _call(theirs, x, 5, 64)
    _call(theirs, x, 5, 64)
    assert flydsl_paths.compiled == [] and [a for _, a, _ in flydsl_paths.own] == [(x, 5, 64)] * 2


def test_an_argument_flydsl_cannot_sign_keeps_flydsls_path(launch, flydsl_paths):
    _call(launch, bf16(4, 8), None, 64)
    assert flydsl_paths.compiled == [] and len(flydsl_paths.own) == 1


def test_a_call_flyc_compile_would_not_launch_keeps_flydsls_path(launch, flydsl_paths, monkeypatch):
    monkeypatch.setattr(flydsl_cache, "_would_launch", lambda values: False)
    _call(launch, bf16(4, 8), 5, 64)
    assert flydsl_paths.compiled == [] and len(flydsl_paths.own) == 1


def test_a_tensor_off_the_device_is_a_placeholder_flyc_compile_does_not_launch():
    assert not flydsl_cache._would_launch((bf16(4, 8), 5, 64, 1))


def test_a_compile_only_process_keeps_nothing(launch, flydsl_paths, monkeypatch):
    monkeypatch.setattr(flydsl_jit, "_compile_impl", lambda jf, *a, **k: flydsl_paths.compiled.append(a))
    _call(launch, bf16(4, 8), 5, 64)
    _call(launch, bf16(4, 8), 5, 64)
    assert len(flydsl_paths.compiled) == 2


def test_a_call_inside_a_trace_is_inlined_by_flydsls_path(launch, flydsl_paths, monkeypatch):
    assert not flydsl_cache._tracing()
    monkeypatch.setattr(flydsl_cache, "_tracing", lambda: True)
    _call(launch, bf16(4, 8), 5, 64)
    assert flydsl_paths.compiled == [] and len(flydsl_paths.own) == 1


def test_install_wraps_jit_function_calls_once(monkeypatch):
    monkeypatch.setattr(flydsl_jit.JitFunction, "__call__", flydsl_jit.JitFunction.__call__)
    monkeypatch.setattr(flydsl_cache, "_original_call", None)
    original = flydsl_jit.JitFunction.__call__
    assert flydsl_cache.install() and flydsl_cache.install()
    assert flydsl_jit.JitFunction.__call__.__wrapped__ is original
    assert flydsl_cache._original_call is original


@pytest.mark.parametrize(("version", "needed"), [("0.4.2", False), ("0.5.0", True), ("0.5.1.dev3", True)])
def test_only_a_flydsl_that_traces_every_call_needs_it(monkeypatch, version, needed):
    monkeypatch.setattr(flydsl, "__version__", version)
    assert flydsl_cache.is_needed() is needed
