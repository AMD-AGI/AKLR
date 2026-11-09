# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin switch, the routes, and falling back to aiter."""

import importlib
import importlib.metadata
import inspect
import pkgutil
import sys
from types import ModuleType

import pytest
import torch

import aklr.plugins.vllm as plugin
from aklr.core.platform.env import ENV_DISABLE, ENV_VLLM_PLUGIN
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm import hook, warmup

# at collection, as the plugin does in each worker: before any test empties hook._ROUTES
importlib.import_module("aklr.plugins.vllm.layers")
importlib.import_module("aklr.plugins.vllm.models")
DECLARED = list(hook._ROUTES)
INSTALLED = "0.33.0rc1.dev910"


def test_no_two_routes_take_one_callee():
    callees = [callee for _, callees, _, _ in DECLARED for callee in callees]
    assert callees and len(set(callees)) == len(callees)


def installed_vllm():
    try:
        return hook.vllm_version()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vllm is not installed")


def test_every_route_was_tested_on_the_installed_vllm():
    """A new vLLM image needs the routes adapted."""
    installed = installed_vllm()
    assert [caller for caller, _, versions, _ in DECLARED if installed not in versions] == []


def parameters(fn):
    # aiter's defaults are aiter's enums, which api modules cannot import
    return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize(
    ("callee", "route"),
    [(c, r) for _, callees, _, r in DECLARED for c in callees],
    ids=[c for _, callees, _, _ in DECLARED for c in callees],
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
    assert hook.vllm_version() == release


@pytest.fixture
def fake(monkeypatch):
    """No route declared; a module ``fake_aiter`` with ``callee``, and ``caller`` that looks
    it up when it runs."""
    module = ModuleType("fake_aiter")
    module.callee = lambda x: f"aiter({x})"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.setattr(hook, "vllm_version", lambda: INSTALLED)
    return module


def caller(x):
    from fake_aiter import callee

    return callee(x)


CALLER = f"{__name__}.caller"


def declare(route, versions=(INSTALLED,)):
    hook.vllm_route(CALLER, "fake_aiter.callee", versions=versions)(route)


def test_a_route_serves(fake):
    declare(lambda x: f"aklr({x})")
    hook.install()
    assert caller(1) == "aklr(1)"


def test_a_fallback_reason_runs_aiter(fake):
    declare(lambda x: FallbackReason.NOT_MEASURED)
    hook.install()
    assert caller(1) == "aiter(1)"


def test_a_raising_route_raises(fake):
    def boom(x):
        raise RuntimeError("boom")

    declare(boom)
    hook.install()
    with pytest.raises(RuntimeError, match="boom"):
        caller(1)


def test_installing_twice_wraps_once(fake):
    declare(lambda x: FallbackReason.NOT_MEASURED)
    hook.install()
    hook.install()
    assert fake.callee.__wrapped__(1) == "aiter(1)"
    assert not hasattr(fake.callee.__wrapped__, "aklr_route")


def test_a_route_untested_on_the_installed_vllm_is_not_hooked(fake):
    declare(lambda x: f"aklr({x})", versions=("0.32.0",))
    hook.install()
    assert caller(1) == "aiter(1)"


def test_a_caller_that_does_not_call_the_callee_raises(fake):
    hook.vllm_route(f"{__name__}.test_a_route_serves", "fake_aiter.callee", versions=(INSTALLED,))(
        lambda x: x
    )
    with pytest.raises(RuntimeError, match="does not call"):
        hook.install()


def test_inside_a_compiled_region_the_route_is_not_traced(fake):
    traced = []
    fake.callee = lambda x: x * 2
    declare(lambda x: traced.append(x) or x + 10)
    hook.install()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(lambda x: fake.callee(x), backend="eager")
        assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))
    finally:
        torch._dynamo.reset()
    assert traced == []
    assert torch.equal(fake.callee(torch.ones(2)), torch.full((2,), 11.0))


@pytest.fixture
def worker(monkeypatch):
    """A fake ``vllm.v1.worker.gpu_worker.Worker``, and what register() and its first
    ``init_device`` did, in order."""
    calls = []

    class Worker:
        def init_device(self):
            calls.append("device bound")

        def compile_or_warm_up_model(self):
            calls.append("vllm warms up")

        def execute_model(self, step):
            calls.append(step)
            return step

    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(warmup, "warm_up", lambda worker: calls.append("aklr warms up"))
    monkeypatch.setattr(plugin, "register_vllm_plugin", lambda: calls.append("aklr starts"))
    monkeypatch.setattr(plugin, "op_packages", lambda: ("aklr.ops.gemm", "aklr.plugins.vllm.x.y"))
    monkeypatch.setattr(plugin.importlib, "import_module", calls.append)
    monkeypatch.setattr(hook, "install", lambda: calls.append("install"))
    plugin._install_once.cache_clear()
    yield Worker, calls
    plugin._install_once.cache_clear()


@pytest.mark.parametrize(
    "env, hooked",
    [
        ({}, False),
        ({ENV_VLLM_PLUGIN: "0"}, False),
        ({ENV_VLLM_PLUGIN: "1"}, True),
        ({ENV_VLLM_PLUGIN: "1", ENV_DISABLE: "1"}, False),
    ],
)
def test_only_1_turns_the_routes_on(monkeypatch, worker, env, hooked):
    """The routes hook only with AKLR_VLLM_PLUGIN=1, not with AKLR_DISABLE=1, and only in a
    worker, once its device is bound; AKLR warms up after vLLM does."""
    Worker, calls = worker
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    plugin.register()
    assert calls == ["aklr starts"]
    Worker().init_device()
    Worker().init_device()
    Worker().compile_or_warm_up_model()
    routes = (
        [
            "aklr.plugins.vllm.layers",
            "aklr.plugins.vllm.models",
            "aklr.plugins.vllm.x.y.api",
            "install",
        ]
        if hooked
        else []
    )
    warmed = ["aklr warms up"] if hooked else []
    assert calls == ["aklr starts", "device bound", *routes, "device bound", "vllm warms up", *warmed]
    assert Worker().execute_model(step="step") == "step"
    assert calls[-1] == "step"


def test_any_other_switch_value_ends_the_process(monkeypatch, worker):
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "true")
    with pytest.raises(SystemExit, match=ENV_VLLM_PLUGIN):
        plugin.register()
