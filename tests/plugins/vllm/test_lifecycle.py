# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The real core/plugin hook composition, using a CPU-only fake Worker."""

import functools
import sys
from types import ModuleType, SimpleNamespace

import pytest

import aklr.plugins.vllm as plugin
from aklr.core.runtime.dispatch import DispatcherRegistry, register_vllm_plugin
from aklr.core.runtime.hook_registry import HookRegistry, HookSpec


@pytest.fixture
def lifecycle(monkeypatch):
    events = []
    registry = HookRegistry()
    monkeypatch.setattr(HookRegistry, "default", classmethod(lambda cls: registry))

    class Worker:
        fail_init = False

        def init_device(self):
            events.append("device")
            if self.fail_init:
                raise RuntimeError("device failed")
            return "bound"

        def compile_or_warm_up_model(self):
            events.append("native warmup")
            return 42

        def execute_model(self, step):
            events.append("execute")
            return step

    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = Worker
    module.compute = lambda value: value * 2
    monkeypatch.setitem(sys.modules, module.__name__, module)

    @functools.cache
    def init_once():
        events.append("runtime")

    monkeypatch.setattr(
        DispatcherRegistry, "default", classmethod(lambda cls: SimpleNamespace(init_once=init_once))
    )

    @functools.cache
    def install():
        assert "runtime" in events
        events.append("routes")
        registry.register(
            HookSpec(
                "test.compute",
                f"{module.__name__}.compute",
                lambda next_fn, value: next_fn(value) + 1,
            )
        )
        return registry.apply("device_ready")

    monkeypatch.setattr(plugin, "_install_once", install)
    from aklr.plugins.vllm import warmup

    monkeypatch.setattr(warmup, "warm_up", lambda worker: events.append("aklr warmup"))

    def roomy(next_fn, *args, **kwargs):
        events.append("roomy")
        return next_fn(*args, **kwargs)

    monkeypatch.setattr(plugin, "roomy", roomy)
    try:
        yield Worker, module, registry, events
    finally:
        registry.restore()


def test_bootstrap_and_device_ready_are_ordered_and_idempotent(lifecycle, monkeypatch):
    Worker, module, registry, events = lifecycle
    monkeypatch.setenv("AKLR_VLLM_PLUGIN", "1")
    plugin.register()
    installed = Worker.init_device
    plugin.register()
    assert Worker.init_device is installed
    assert events == []
    assert {row["phase"] for row in registry.report()} == {"bootstrap"}

    worker = Worker()
    assert worker.init_device() == "bound"
    assert worker.init_device() == "bound"
    assert events == ["device", "runtime", "routes", "device"]
    assert module.compute(3) == 7
    assert {row["phase"] for row in registry.report()} == {"bootstrap", "device_ready"}
    assert worker.compile_or_warm_up_model() == 42
    assert events[-2:] == ["native warmup", "aklr warmup"]
    assert worker.execute_model("step") == "step"
    assert events[-2:] == ["roomy", "execute"]


@pytest.mark.parametrize("first", [register_vllm_plugin, plugin.register])
@pytest.mark.parametrize("initialize_before_second", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_public_registration_order_preserves_one_bootstrap_batch(
    lifecycle, monkeypatch, first, initialize_before_second, enabled
):
    Worker, module, registry, events = lifecycle
    monkeypatch.setenv("AKLR_VLLM_PLUGIN", "1" if enabled else "0")
    second = plugin.register if first is register_vllm_plugin else register_vllm_plugin
    methods = ("init_device", "compile_or_warm_up_model", "execute_model")
    originals = {name: getattr(Worker, name) for name in methods}
    first()
    installed = {name: getattr(Worker, name) for name in methods}
    assert installed["init_device"] is not originals["init_device"]
    for name in methods[1:]:
        assert (installed[name] is not originals[name]) is enabled
    worker = Worker()
    if initialize_before_second:
        assert worker.init_device() == "bound"

    # Both public entrances stay idempotent even after the device hooks have run.
    second()
    first()
    second()
    for name in methods:
        assert getattr(Worker, name) is installed[name]
    if not initialize_before_second:
        assert events == []
    assert worker.init_device() == "bound"
    expected = ["device", "runtime"]
    if enabled:
        expected.append("routes")
    if initialize_before_second:
        expected.append("device")
    assert events == expected
    assert module.compute(3) == (7 if enabled else 6)
    bootstrap_ids = {row["hook_id"] for row in registry.report() if row["phase"] == "bootstrap"}
    assert bootstrap_ids == {"vllm.runtime.init"} | (
        {"vllm.plugin.routes", "vllm.plugin.warmup", "vllm.plugin.execute"} if enabled else set()
    )
    assert worker.compile_or_warm_up_model() == 42
    assert events[-(2 if enabled else 1) :] == ["native warmup", *(["aklr warmup"] if enabled else [])]
    assert worker.execute_model("step") == "step"
    assert events[-(2 if enabled else 1) :] == [*(["roomy"] if enabled else []), "execute"]


@pytest.mark.parametrize("register", [register_vllm_plugin, plugin.register])
def test_missing_plugin_target_keeps_core_bootstrap_uninstalled(lifecycle, monkeypatch, register):
    Worker, _, registry, events = lifecycle
    monkeypatch.setenv("AKLR_VLLM_PLUGIN", "1")
    original_init, original_warmup = Worker.init_device, Worker.compile_or_warm_up_model
    del Worker.execute_model
    with pytest.raises((RuntimeError, SystemExit), match="execute_model"):
        register()
    assert Worker.init_device is original_init
    assert Worker.compile_or_warm_up_model is original_warmup
    assert all(row["state"] == "registered" for row in registry.report())
    assert events == []


@pytest.mark.parametrize("env", [{}, {"AKLR_VLLM_PLUGIN": "0"}, {"AKLR_DISABLE": "1"}])
def test_disabled_routes_preserve_core_initialization(lifecycle, monkeypatch, env):
    Worker, module, registry, events = lifecycle
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    plugin.register()
    assert Worker().init_device() == "bound"
    assert events == ["device", "runtime"]
    assert module.compute(3) == 6
    assert [row["hook_id"] for row in registry.report()] == ["vllm.runtime.init"]


def test_failed_device_does_not_initialize_runtime_or_routes(lifecycle, monkeypatch):
    Worker, module, registry, events = lifecycle
    monkeypatch.setenv("AKLR_VLLM_PLUGIN", "1")
    plugin.register()
    Worker.fail_init = True
    with pytest.raises(RuntimeError, match="device failed"):
        Worker().init_device()
    assert events == ["device"]
    assert module.compute(3) == 6
    assert all(row["phase"] == "bootstrap" for row in registry.report())


@pytest.mark.parametrize("register", [register_vllm_plugin, plugin.register])
def test_invalid_switch_does_not_patch_worker(lifecycle, monkeypatch, register):
    Worker, _, registry, events = lifecycle
    original = Worker.init_device
    monkeypatch.setenv("AKLR_VLLM_PLUGIN", "true")
    error = ValueError if register is register_vllm_plugin else SystemExit
    with pytest.raises(error, match="AKLR_VLLM_PLUGIN"):
        register()
    assert Worker.init_device is original
    assert registry.report() == ()
    assert events == []
