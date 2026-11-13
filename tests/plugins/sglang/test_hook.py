# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin switch, the routes, and falling back to sglang."""

import importlib
import importlib.metadata
import inspect
import pkgutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang as plugin
from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import ENV_DISABLE, ENV_SGLANG_PLUGIN, ENV_TRACE
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import hook

# at collection, as the plugin does at start-up: before any test empties hook._ROUTES
for package in op_packages():
    if package.startswith("aklr.plugins.sglang."):
        importlib.import_module(f"{package}.api")
importlib.import_module("aklr.plugins.sglang.aiter_calls")
DECLARED = list(hook._ROUTES)
INSTALLED = "0.5.22"


def test_no_two_routes_take_one_target():
    targets = [target for target, _, _ in DECLARED]
    assert targets and len(set(targets)) == len(targets)


def installed_sglang():
    try:
        return hook.sglang_version()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang is not installed")


def test_every_route_was_tested_on_the_installed_sglang():
    """A new sglang image needs the routes adapted."""
    installed = installed_sglang()
    assert [target for target, versions, _ in DECLARED if installed not in versions] == []


def parameters(fn, *, defaults=True):
    return [
        (p.name, p.kind, *((p.default,) if defaults else ()))
        for p in inspect.signature(fn).parameters.values()
    ]


@pytest.mark.parametrize(
    ("target", "route"), [(t, r) for t, _, r in DECLARED], ids=[t for t, _, _ in DECLARED]
)
def test_every_route_takes_its_targets_parameters(target, route):
    installed_sglang()
    if not target.startswith("aiter."):
        assert parameters(route) == parameters(pkgutil.resolve_name(target))
        return
    # aiter's defaults are aiter's enums, which route modules cannot import. A newer aiter than the
    # installed one adds parameters at the end, which the route takes too.
    taken = parameters(pkgutil.resolve_name(target), defaults=False)
    assert parameters(route, defaults=False)[: len(taken)] == taken


def test_the_fused_moe_route_passes_sglangs_output_to_the_op(monkeypatch):
    """CI's aiter lacks ``output``, so the parameter test cannot see it: the route must take
    sglang's ``output`` (MoRI EP's combine buffer) and pass it on."""
    from aklr.plugins.sglang import aiter_calls

    passed = {}
    monkeypatch.setattr(aiter_calls.aklr_aiter, "fused_moe", lambda *args, **kwargs: passed.update(kwargs))
    out = torch.empty(0)
    aiter_calls._fused_moe(*[None] * 5, output=out)
    assert passed["output"] is out


@pytest.mark.parametrize(
    ("installed", "release"),
    [("0.5.22+rocm7.0", "0.5.22"), ("0.5.22.dev20260906+g5bebe7a", "0.5.22.dev20260906")],
)
def test_only_a_local_label_is_dropped(monkeypatch, installed, release):
    """A development build is not the release it precedes."""
    monkeypatch.setattr(hook, "version", lambda dist: installed)
    assert hook.sglang_version() == release


def original(x):
    return f"sglang({x})"


TARGET = f"{__name__}.original"


@pytest.fixture
def hooks(monkeypatch):
    """No route declared; what install() hooks, as sglang's HookRegistry records it."""
    registered = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = SimpleNamespace(AROUND="around")
    registry.HookRegistry = SimpleNamespace(register=lambda *args: registered.append(args))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.setattr(hook, "sglang_version", lambda: INSTALLED)
    return registered


def declare(route):
    hook.sglang_route(TARGET, versions=(INSTALLED,))(route)


def wrapper(hooks):
    (around,) = [w for t, w, kind in hooks if t == TARGET and kind == "around"]
    return around


def test_a_route_serves(hooks):
    declare(lambda x: f"aklr({x})")
    hook.install()
    assert wrapper(hooks)(original, 1) == "aklr(1)"


def test_a_fallback_reason_runs_sglang(hooks):
    declare(lambda x: FallbackReason.NOT_MEASURED)
    hook.install()
    assert wrapper(hooks)(original, 1) == "sglang(1)"


@pytest.mark.parametrize(
    ("mode", "is_traced"), [("0", False), ("op", False), ("plugin", True), ("all", True)]
)
def test_trace_names_the_route_and_its_result_once(hooks, monkeypatch, caplog, mode, is_traced):
    monkeypatch.setenv(ENV_TRACE, mode)
    route = lambda x: f"aklr({x})" if x else FallbackReason.NOT_MEASURED
    declare(route)
    hook.install()
    around = wrapper(hooks)
    assert [around(original, x) for x in (1, 1, 0, 0)] == ["aklr(1)", "aklr(1)", "sglang(0)", "sglang(0)"]
    lines = [r.message for r in caplog.records if r.name == "aklr.trace.plugin"]
    if not is_traced:
        assert lines == []
        return
    prefix = f"framework=sglang target={TARGET} route={route.__module__}.{route.__qualname__}"
    assert [line.split(" ", 3)[3] for line in lines] == [
        f"{prefix} result=served reason=-",
        f"{prefix} result=fallback reason=not_measured",
    ]
    assert all(line.startswith("[aklr plugin] pid=") for line in lines)


def test_a_raising_route_raises(hooks):
    def boom(x):
        raise RuntimeError("boom")

    declare(boom)
    hook.install()
    with pytest.raises(RuntimeError, match="boom"):
        wrapper(hooks)(original, 1)


def test_a_route_untested_on_the_installed_sglang_is_neither_resolved_nor_hooked(hooks):
    hook.sglang_route(f"{__name__}.missing", versions=("0.5.21",))(lambda x: x)
    hook.install()
    assert hooks == []


def test_a_target_that_does_not_resolve_raises(hooks):
    hook.sglang_route(f"{__name__}.missing", versions=(INSTALLED,))(lambda x: x)
    with pytest.raises((AttributeError, ImportError)):
        hook.install()


def test_inside_a_compiled_region_the_route_is_not_traced(hooks):
    traced = []
    declare(lambda x: traced.append(x) or x + 10)
    hook.install()
    around = wrapper(hooks)

    def sglangs(x):
        return x * 2

    torch._dynamo.reset()
    try:
        compiled = torch.compile(lambda x: around(sglangs, x), backend="eager")
        assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))
    finally:
        torch._dynamo.reset()
    assert traced == []
    assert torch.equal(around(sglangs, torch.ones(2)), torch.full((2,), 11.0))


@pytest.fixture
def register_calls(monkeypatch):
    """What register() did, in order."""
    calls = []
    monkeypatch.setattr(plugin, "register_sglang_plugin", lambda: calls.append("aklr starts"))
    monkeypatch.setattr(plugin, "op_packages", lambda: ("aklr.ops.gemm", "aklr.plugins.sglang.x.y"))
    monkeypatch.setattr(plugin.importlib, "import_module", calls.append)
    monkeypatch.setattr(hook, "install", lambda: calls.append("install"))
    return calls


@pytest.mark.parametrize(
    "env, hooked",
    [
        ({}, False),
        ({ENV_SGLANG_PLUGIN: "0"}, False),
        ({ENV_SGLANG_PLUGIN: "1"}, True),
        ({ENV_SGLANG_PLUGIN: "1", ENV_DISABLE: "1"}, False),
    ],
)
def test_only_1_turns_the_routes_on(monkeypatch, register_calls, env, hooked):
    """The layer routes hook only with AKLR_SGLANG_PLUGIN=1, and not with AKLR_DISABLE=1."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    plugin.register()
    routes = (
        [
            "aklr.plugins.sglang.x.y.api",
            "aklr.plugins.sglang.aiter_calls",
            "install",
        ]
        if hooked
        else []
    )
    assert register_calls == ["aklr starts", *routes]


def test_any_other_switch_value_ends_the_process(monkeypatch, register_calls):
    monkeypatch.setenv(ENV_SGLANG_PLUGIN, "true")
    with pytest.raises(SystemExit, match=ENV_SGLANG_PLUGIN):
        plugin.register()


def test_a_plugin_that_cannot_start_ends_the_process(monkeypatch, register_calls):
    """sglang only logs an exception from a plugin."""
    monkeypatch.setenv(ENV_SGLANG_PLUGIN, "1")

    def install():
        raise ValueError("broken declaration")

    monkeypatch.setattr(hook, "install", install)
    with pytest.raises(SystemExit, match="broken declaration"):
        plugin.register()
