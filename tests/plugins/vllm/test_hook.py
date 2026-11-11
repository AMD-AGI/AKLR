# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin switch, the routes, and falling back to aiter."""

import importlib
import importlib.metadata
import inspect
import pkgutil
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
import torch

import aklr.plugins.vllm as plugin
from aklr.core.platform.env import ENV_DISABLE, ENV_VLLM_PLUGIN
from aklr.core.runtime._vllm_hook_registry import HookRegistry
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.plugins.vllm import hook, warmup

# at collection, as the plugin does in each worker: before any test empties hook._ROUTES
importlib.import_module("aklr.plugins.vllm.layers")
importlib.import_module("aklr.plugins.vllm.models")
DECLARED = [
    (callee, route, declaration)
    for route, targets in hook._ROUTES.items()
    for callee, declaration in targets.items()
]
INSTALLED = "0.33.0rc1.dev910"


@pytest.fixture(autouse=True)
def hook_registry(monkeypatch):
    """Each test owns its bindings and leaves the process's default registry untouched."""
    registry = HookRegistry()
    monkeypatch.setattr(HookRegistry, "default", classmethod(lambda cls: registry))
    try:
        yield registry
    finally:
        registry.restore()


def test_the_entry_point_is_the_plugins():
    pyproject = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())
    entry = pyproject["project"]["entry-points"]["vllm.general_plugins"]["aklr"]
    assert pkgutil.resolve_name(entry.replace(":", ".")) is plugin.register


def test_no_two_routes_take_one_callee():
    callees = [callee for callee, _, _ in DECLARED]
    assert callees and len(set(callees)) == len(callees)


def installed_vllm():
    try:
        return hook._vllm_version()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vllm is not installed")


def test_every_route_was_tested_on_the_installed_vllm():
    """A new vLLM image needs the routes adapted."""
    installed = installed_vllm()
    assert [d.caller for _, _, d in DECLARED if installed not in d.versions] == []


def parameters(fn):
    # aiter's defaults are aiter's enums, which api modules cannot import
    return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize(
    ("callee", "route"),
    [(callee, route) for callee, route, _ in DECLARED],
    ids=[callee for callee, _, _ in DECLARED],
)
def test_every_route_takes_its_callees_parameters(callee, route):
    installed_vllm()
    assert parameters(route) == parameters(pkgutil.resolve_name(callee))


@pytest.mark.parametrize(
    ("installed", "release"),
    [("0.33.0rc1.dev910+g4fab5a074", "0.33.0rc1.dev910"), ("0.33.0+rocm7.0", "0.33.0")],
)
def test_only_a_local_label_is_dropped(monkeypatch, installed, release):
    """A development build is not the release it precedes."""
    monkeypatch.setattr(hook, "version", lambda dist: installed)
    assert hook._vllm_version() == release


@pytest.fixture
def fake(monkeypatch):
    """No route declared; a module ``fake_aiter`` with ``callee``, and ``caller`` that looks
    it up when it runs."""
    module = ModuleType("fake_aiter")
    module.callee = lambda x: f"aiter({x})"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(hook, "_ROUTES", {})
    monkeypatch.setattr(hook, "_vllm_version", lambda: INSTALLED)
    return module


def caller(x):
    from fake_aiter import callee

    return callee(x)


CALLER = f"{__name__}.caller"


def declare(route, versions=(INSTALLED,), group=None):
    hook.vllm_route(CALLER, "fake_aiter.callee", versions=versions, group=group)(route)


def test_a_route_serves(fake):
    declare(lambda x: f"aklr({x})")
    hook._install()
    assert caller(1) == "aklr(1)"


def test_a_fallback_reason_runs_aiter(fake):
    declare(lambda x: FallbackReason.NOT_MEASURED)
    hook._install()
    assert caller(1) == "aiter(1)"


def test_a_raising_route_raises(fake):
    def boom(x):
        raise RuntimeError("boom")

    declare(boom)
    hook._install()
    with pytest.raises(RuntimeError, match="boom"):
        caller(1)


def test_installing_twice_wraps_once(fake):
    native = fake.callee
    declare(lambda x: FallbackReason.NOT_MEASURED)
    hook._install()
    wrapped = fake.callee
    hook._install()
    assert fake.callee is wrapped
    assert wrapped.__wrapped__ is native
    assert caller(1) == "aiter(1)"


def test_a_route_untested_on_the_installed_vllm_is_not_hooked(fake):
    declare(lambda x: f"aklr({x})", versions=("0.32.0",))
    hook._install()
    assert caller(1) == "aiter(1)"


def test_a_caller_that_does_not_call_the_callee_raises(fake):
    hook.vllm_route(f"{__name__}.test_a_route_serves", "fake_aiter.callee", versions=(INSTALLED,))(
        lambda x: x
    )
    with pytest.raises(RuntimeError, match="does not call"):
        hook._install()


def test_inside_a_compiled_region_the_route_is_not_traced(fake):
    traced = []
    fake.callee = lambda x: x * 2
    declare(lambda x: traced.append(x) or x + 10)
    hook._install()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(lambda x: fake.callee(x), backend="eager")
        assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))
    finally:
        torch._dynamo.reset()
    assert traced == []
    assert torch.equal(fake.callee(torch.ones(2)), torch.full((2,), 11.0))


def other_caller(x):
    from fake_aiter import other

    return other(x)


@pytest.mark.parametrize("other_versions", [(INSTALLED,), ("0.32.0",)])
def test_group_members_are_installed_or_skipped_together(fake, other_versions):
    fake.other = lambda x: f"other({x})"
    declare(lambda x: f"aklr({x})", group="paired")
    hook.vllm_route(f"{__name__}.other_caller", "fake_aiter.other", versions=other_versions, group="paired")(
        lambda x: f"aklr_other({x})"
    )
    installed = hook._install()
    if INSTALLED in other_versions:
        assert set(installed) == {"fake_aiter.callee", "fake_aiter.other"}
        assert (caller(1), other_caller(1)) == ("aklr(1)", "aklr_other(1)")
    else:
        assert installed == []
        assert (caller(1), other_caller(1)) == ("aiter(1)", "other(1)")


def test_invalid_group_member_leaves_every_binding_native(fake):
    fake.other = lambda x: f"other({x})"
    native, other_native = fake.callee, fake.other

    def route(x):
        return f"aklr({x})"

    declare(route, group="paired")
    hook.vllm_route(f"{__name__}.other_caller", "fake_aiter.other", versions=(INSTALLED,), group="paired")(
        lambda x, extra: x
    )
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        hook._install()
    assert fake.callee is native
    assert fake.other is other_native
    assert (caller(1), other_caller(1)) == ("aiter(1)", "other(1)")
    with pytest.raises(RuntimeError, match="has not been applied"):
        hook._original(route)


@pytest.fixture
def worker(monkeypatch, fake):
    """A complete fake vLLM module hierarchy, retaining the real plugin bootstrap."""
    calls = []

    class Worker:
        def init_device(self):
            calls.append("device bound")
            return "bound"

        def compile_or_warm_up_model(self):
            calls.append("vllm warms up")
            return 1.25

        def execute_model(self, step):
            calls.append(step)
            return step

    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    module.Worker = Worker
    monkeypatch.setattr(warmup, "warm_up", lambda worker: calls.append("aklr warms up"))
    monkeypatch.setattr(DispatcherRegistry, "init_once", lambda self: calls.append("aklr starts"))
    return Worker, calls


@pytest.mark.parametrize(
    "env, hooked",
    [
        ({}, False),
        ({ENV_VLLM_PLUGIN: "0"}, False),
        ({ENV_VLLM_PLUGIN: "1"}, True),
        ({ENV_VLLM_PLUGIN: "1", ENV_DISABLE: "1"}, False),
    ],
)
def test_only_1_turns_the_routes_on(monkeypatch, worker, fake, env, hooked):
    """The routes hook only with AKLR_VLLM_PLUGIN=1, not with AKLR_DISABLE=1, and only in a
    worker, once its device is bound; AKLR warms up after vLLM does."""
    Worker, calls = worker
    declare(lambda x: f"aklr({x})")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    plugin.register()
    assert calls == []
    init_device = Worker.init_device
    plugin.register()
    assert Worker.init_device is init_device
    assert Worker().init_device() == "bound"
    assert caller(1) == ("aklr(1)" if hooked else "aiter(1)")
    installed = fake.callee
    assert Worker().init_device() == "bound"
    assert fake.callee is installed
    assert Worker().compile_or_warm_up_model() == 1.25
    warmed = ["aklr warms up"] if hooked else []
    assert calls == ["device bound", "aklr starts", "device bound", "aklr starts", "vllm warms up", *warmed]
    assert Worker().execute_model(step="step") == "step"
    assert calls[-1] == "step"


def test_no_warmup_where_no_route_hooked(monkeypatch, worker):
    """On a vLLM no route was tested on, AKLR leaves vLLM's model runner alone."""
    Worker, calls = worker
    declare(lambda x: f"aklr({x})", versions=("0.32.0",))
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "1")
    plugin.register()
    Worker().init_device()
    assert Worker().compile_or_warm_up_model() == 1.25
    assert caller(1) == "aiter(1)"
    assert calls == ["device bound", "aklr starts", "vllm warms up"]


@pytest.mark.parametrize("phase", [None, "device_ready"], ids=["all", "routes_only"])
def test_restored_routes_are_reinstalled(monkeypatch, worker, fake, hook_registry, phase):
    """A new init after cleanup installs routes and captures the current native callable."""
    Worker, _ = worker
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "1")

    def route(x):
        return f"aklr({x})"

    declare(route)
    native = fake.callee
    plugin.register()
    Worker().init_device()
    assert caller(1) == "aklr(1)"
    assert hook._original(route) is native

    hook_registry.restore(phase)
    assert fake.callee is native
    with pytest.raises(RuntimeError, match="has not been applied"):
        hook._original(route)

    def next_native(x):
        return f"next_aiter({x})"

    fake.callee = next_native
    plugin.register()
    Worker().init_device()
    assert caller(1) == "aklr(1)"
    assert hook._original(route) is next_native
    wrapped = fake.callee
    assert wrapped.__wrapped__ is next_native
    Worker().init_device()
    assert fake.callee is wrapped


def test_any_other_switch_value_ends_the_process(monkeypatch, worker):
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "true")
    with pytest.raises(SystemExit, match=ENV_VLLM_PLUGIN):
        plugin.register()
