# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The graph-capture tripwire: which compiles it catches on each FlyDSL, with stand-in compilers."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from aklr.core.jit import flydsl_cache
from aklr.core.validation.stages._bench._evaluation._candidate import _CompileTripwire


class _Compiler:
    def compile(self, *args):
        return "compiled"


@pytest.fixture
def compilers(monkeypatch):
    """Stand-ins for Triton's and FlyDSL's compile entry points, each answering when not tripped."""
    triton_jit = SimpleNamespace(
        JITFunction=type("JITFunction", (), {"_do_compile": lambda self: "compiled"})
    )
    flydsl_jit = SimpleNamespace(MlirCompiler=_Compiler, _build_call_state=lambda *args: "built")
    for name, module in {
        "triton": SimpleNamespace(runtime=SimpleNamespace(jit=triton_jit)),
        "triton.runtime": SimpleNamespace(jit=triton_jit),
        "triton.runtime.jit": triton_jit,
        "flydsl": SimpleNamespace(compiler=SimpleNamespace(jit_function=flydsl_jit)),
        "flydsl.compiler": SimpleNamespace(jit_function=flydsl_jit),
        "flydsl.compiler.jit_function": flydsl_jit,
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(vars(module))
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(flydsl_cache, "_compile", lambda *args: "compiled")
    return SimpleNamespace(
        triton=sys.modules["triton.runtime.jit"], flydsl=sys.modules["flydsl.compiler.jit_function"]
    )


@pytest.mark.parametrize(
    ("cache_needed", "flydsl_hooks"),
    [(True, {"_compile", "compile"}), (False, {"_build_call_state"})],
    ids=["flydsl_0.5_retraces", "flydsl_before_0.5"],
)
def test_the_tripwire_hooks_the_compiles_of_this_flydsl(monkeypatch, compilers, cache_needed, flydsl_hooks):
    monkeypatch.setattr(flydsl_cache, "is_needed", lambda: cache_needed)
    tripwire = _CompileTripwire.resolve()
    assert {name for _, name in tripwire.hooks} == {"_do_compile", *flydsl_hooks}


def test_an_armed_tripwire_trips_on_each_hook_and_restores_them(monkeypatch, compilers):
    monkeypatch.setattr(flydsl_cache, "is_needed", lambda: True)
    tripwire = _CompileTripwire.resolve()
    before = [getattr(owner, name) for owner, name in tripwire.hooks]
    with tripwire.armed() as compile_calls:
        for owner, name in tripwire.hooks:
            with pytest.raises(RuntimeError, match=f"{name}: compiled in the capture"):
                getattr(owner, name)()
    assert compile_calls == [name for _, name in tripwire.hooks]
    assert [getattr(owner, name) for owner, name in tripwire.hooks] == before
    # disarmed, a compile runs again
    assert compilers.flydsl.MlirCompiler().compile() == "compiled"
    assert flydsl_cache._compile() == "compiled"
