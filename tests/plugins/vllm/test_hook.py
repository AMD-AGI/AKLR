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
from types import ModuleType, SimpleNamespace

import pytest
import torch

import aklr.plugins.vllm as plugin
from aklr.core.platform.env import ENV_DISABLE, ENV_VLLM_PLUGIN
from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.runtime.hook_registry import HookRegistry
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
def registry(monkeypatch):
    registry = HookRegistry()
    monkeypatch.setattr(HookRegistry, "default", staticmethod(lambda: registry))
    monkeypatch.setattr(hook, "_ROUTES", {route: dict(targets) for route, targets in hook._ROUTES.items()})
    yield registry
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
        return hook.vllm_version()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vllm is not installed")


def test_every_route_was_tested_on_the_installed_vllm():
    """A new vLLM image needs the routes adapted."""
    installed = installed_vllm()
    assert [
        declaration.caller for _, _, declaration in DECLARED if installed not in declaration.versions
    ] == []


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
    assert hook.vllm_version() == release


@pytest.fixture
def fake(monkeypatch):
    """No route declared; a module ``fake_aiter`` with ``callee``, and ``caller`` that looks
    it up when it runs."""
    module = ModuleType("fake_aiter")
    module.callee = lambda x: f"aiter({x})"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(hook, "_ROUTES", {})
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


def second_caller(x):
    from fake_aiter import second

    return second(x)


SECOND_CALLER = f"{__name__}.second_caller"


def test_an_incompatible_signature_leaves_the_batch_untouched(fake):
    fake.second = lambda x, extra: (x, extra)
    before = fake.callee
    declare(lambda x: "aklr")
    hook.vllm_route(SECOND_CALLER, "fake_aiter.second", versions=(INSTALLED,))(lambda x: x)
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        hook.install()
    assert fake.callee is before
    assert caller(1) == "aiter(1)"


def test_one_untested_member_skips_the_whole_group(fake, caplog):
    fake.second = lambda x: f"second({x})"
    hook.vllm_route(CALLER, "fake_aiter.callee", versions=(INSTALLED,), group="pair")(lambda x: "aklr first")
    hook.vllm_route(SECOND_CALLER, "fake_aiter.second", versions=("0.32.0",), group="pair")(
        lambda x: "aklr second"
    )
    assert hook.install() == []
    assert caller(1) == "aiter(1)"
    assert second_caller(1) == "second(1)"
    assert f"untested callers: {SECOND_CALLER}" in caplog.text
    assert f"blocked by untested group members: {CALLER} (group=pair)" in caplog.text


def test_multiple_declarations_keep_each_callees_group(fake):
    fake.second = lambda x: f"second({x})"

    def route(x):
        return f"aklr({x})"

    hook.vllm_route(CALLER, "fake_aiter.callee", versions=(INSTALLED,), group="first")(route)
    hook.vllm_route(SECOND_CALLER, "fake_aiter.second", versions=("0.32.0",), group="second")(route)
    assert hook.install() == ["fake_aiter.callee"]
    assert caller(1) == "aklr(1)"
    assert second_caller(1) == "second(1)"


def test_conflicting_declaration_does_not_change_an_existing_route(fake):
    route = lambda x: f"aklr({x})"
    declare(route)
    declare(route)
    with pytest.raises(ValueError, match="conflicting declaration"):
        declare(route, versions=("0.32.0",))
    assert hook.install() == ["fake_aiter.callee"]
    assert caller(1) == "aklr(1)"


def test_saved_original_and_undeclared_alias_remain_native(fake, monkeypatch):
    alias = ModuleType("fake_aiter_importer")
    alias.callee = fake.callee
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    def route(x):
        return f"aklr({x})"

    declare(route)
    hook.install()
    assert caller(1) == "aklr(1)"
    assert alias.callee(1) == "aiter(1)"
    assert hook.original(route) is alias.callee


def test_saved_original_requires_a_declared_installed_target_and_tracks_reinstall(fake, registry):
    first, second = fake.callee, lambda x: f"second({x})"
    fake.second = second

    def route(x):
        return FallbackReason.NOT_MEASURED

    declare(route)
    with pytest.raises(RuntimeError, match="has not been applied"):
        hook.original(route)
    hook.vllm_route(SECOND_CALLER, "fake_aiter.second", versions=(INSTALLED,))(route)
    with pytest.raises(ValueError, match="select one"):
        hook.original(route)
    with pytest.raises(ValueError, match="undeclared callee"):
        hook.original(route, "fake_aiter.unknown")
    with pytest.raises(ValueError, match="not declared"):
        hook.original(lambda x: x)

    hook.install()
    assert hook.original(route, "fake_aiter.callee") is first
    assert hook.original(route, "fake_aiter.second") is second
    registry.restore("device_ready")
    with pytest.raises(RuntimeError, match="has not been applied"):
        hook.original(route, "fake_aiter.callee")

    fake.callee = replacement = lambda x: f"new native({x})"
    hook.install()
    assert hook.original(route, "fake_aiter.callee") is replacement
    assert caller(2) == "new native(2)"
    assert second_caller(2) == "second(2)"


def classmethod_caller(owner, x):
    return owner.callee(x)


def test_classmethod_route_receives_cls_and_preserves_fallback_inheritance_and_restore(fake, registry):
    native_calls = []

    class Owner:
        @classmethod
        def callee(cls, x):
            native_calls.append((cls, x))
            return "native", cls, x

    class Child(Owner):
        pass

    fake.Owner = Owner
    native = inspect.getattr_static(Owner, "callee")

    def route(cls, x):
        if x < 0:
            return FallbackReason.OUT_OF_CONTRACT
        return "aklr", cls, x

    hook.vllm_route(f"{__name__}.classmethod_caller", "fake_aiter.Owner.callee", versions=(INSTALLED,))(route)
    hook.install()
    hook.install()
    assert classmethod_caller(Owner, 1) == ("aklr", Owner, 1)
    assert classmethod_caller(Owner(), 2) == ("aklr", Owner, 2)
    assert classmethod_caller(Child, 3) == ("aklr", Child, 3)
    assert classmethod_caller(Child(), 4) == ("aklr", Child, 4)
    assert native_calls == []
    assert classmethod_caller(Child, -1) == ("native", Child, -1)
    assert native_calls == [(Child, -1)]
    assert hook.original(route) is native.__func__
    assert hook.original(route)(Owner, 5) == ("native", Owner, 5)

    registry.restore("device_ready")
    assert inspect.getattr_static(Owner, "callee") is native
    assert "callee" not in vars(Child)
    assert classmethod_caller(Child(), 6) == ("native", Child, 6)
    with pytest.raises(RuntimeError, match="has not been applied"):
        hook.original(route)


def test_classmethod_route_without_cls_fails_before_any_binding_changes(fake, registry):
    class Owner:
        @classmethod
        def callee(cls, x):
            return "native", cls, x

    fake.Owner = Owner
    native = inspect.getattr_static(Owner, "callee")
    module_native = fake.callee
    declare(lambda x: "aklr")
    hook.vllm_route(f"{__name__}.classmethod_caller", "fake_aiter.Owner.callee", versions=(INSTALLED,))(
        lambda x: "missing cls"
    )
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        hook.install()
    assert inspect.getattr_static(Owner, "callee") is native
    assert fake.callee is module_native
    assert all(row["state"] == "registered" for row in registry.report())


def singleton_gemm_caller(inp, weights, **kwargs):
    from fake_aiter import tgemm

    return tgemm.mm(inp, weights, **kwargs)


def test_tuned_gemm_singleton_route_preserves_bound_original_and_restore(fake, registry):
    """AITER exports TunedGemm() as tgemm; its mm target is bound to that instance."""
    native_calls = []

    class TunedGemm:
        def mm(self, inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
            native_calls.append((self, inp, weights, bias, otype, scale_a, scale_b, scale_c, out))
            return "native", inp, weights, bias, out

    fake.tgemm = TunedGemm()
    other = TunedGemm()
    native_method = TunedGemm.mm

    # This matches the existing _tgemm_mm route's public signature, with no self.
    def route(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
        if scale_a is not None or scale_b is not None or scale_c is not None:
            return FallbackReason.OUT_OF_CONTRACT
        return "aklr", inp, weights, bias, out

    hook.vllm_route(f"{__name__}.singleton_gemm_caller", "fake_aiter.tgemm.mm", versions=(INSTALLED,))(route)
    assert "mm" not in vars(fake.tgemm)
    hook.install()
    hook.install()
    assert "mm" in vars(fake.tgemm)
    assert hook.original(route).__self__ is fake.tgemm
    assert hook.original(route).__func__ is native_method
    assert singleton_gemm_caller(2, 3, bias=4, out="buffer") == ("aklr", 2, 3, 4, "buffer")
    assert not native_calls
    assert singleton_gemm_caller(2, 3, scale_a=5, out="buffer") == ("native", 2, 3, None, "buffer")
    assert native_calls == [(fake.tgemm, 2, 3, None, None, 5, None, None, "buffer")]
    assert other.mm(6, 7) == ("native", 6, 7, None, None)
    assert native_calls[-1][0] is other
    assert TunedGemm.mm is native_method

    registry.restore("device_ready")
    assert "mm" not in vars(fake.tgemm)
    assert fake.tgemm.mm.__func__ is native_method
    assert singleton_gemm_caller(8, 9) == ("native", 8, 9, None, None)
    assert native_calls[-1][0] is fake.tgemm


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

    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(warmup, "warm_up", lambda worker: calls.append("aklr warms up"))
    monkeypatch.setattr(plugin, "_register_runtime_hook", lambda: calls.append("aklr starts"))
    monkeypatch.setattr(plugin, "op_packages", lambda: ("aklr.ops.gemm", "aklr.plugins.vllm.x.y"))
    monkeypatch.setattr(plugin, "importlib", SimpleNamespace(import_module=calls.append))
    hooked = ["aiter.tuned_gemm.tgemm.mm"]
    monkeypatch.setattr(hook, "install", lambda: calls.append("install") or hooked)
    plugin._install_once.cache_clear()
    yield Worker, calls, hooked
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
    Worker, calls, _ = worker
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


def test_no_warmup_where_no_route_hooked(monkeypatch, worker):
    """On a vLLM no route was tested on, AKLR leaves vLLM's model runner alone."""
    Worker, calls, hooked = worker
    hooked.clear()
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "1")
    plugin.register()
    Worker().init_device()
    Worker().compile_or_warm_up_model()
    assert calls[-1] == "vllm warms up"


def test_any_other_switch_value_ends_the_process(monkeypatch, worker):
    monkeypatch.setenv(ENV_VLLM_PLUGIN, "true")
    with pytest.raises(SystemExit, match=ENV_VLLM_PLUGIN):
        plugin.register()
