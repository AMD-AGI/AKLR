# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR_AITER_PLUGIN=capture: AKLR is a provider of aiter's plugin interface that aiter asks only while sglang
captures a decode CUDA graph (its eager warm-ups included), so that eager forwards take aiter's own path; on
sglang's hook registry and AKLR's start stood in, with aiter's plugin interface."""

import pkgutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

import aklr.plugins.sglang as plugin
from aklr.core.platform.env import ENV_AITER_PLUGIN
from aklr.plugins.sglang import hook

CAPTURE = "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape"


@pytest.fixture
def sglang_hooks(monkeypatch):
    """The hooks register() leaves with sglang, by target: [(hook, type)], on the image's sglang."""
    registered = {}
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = SimpleNamespace(AROUND="around", AFTER="after")
    registry.HookRegistry = SimpleNamespace(
        register=lambda target, fn, kind: registered.setdefault(target, []).append((fn, kind))
    )
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    # external distribution metadata and Python resolution, then AKLR's start, not the routing under test
    monkeypatch.setattr(hook, "version", lambda dist: "0.5.23.dev20261120+g28e5763574")
    monkeypatch.setattr(pkgutil, "resolve_name", lambda target: None)
    monkeypatch.setattr(plugin, "register_sglang_plugin", lambda: None)
    # a process's first registration: no route declared yet
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.delitem(sys.modules, "aklr.plugins.sglang.aiter_capture", raising=False)
    return registered


@pytest.fixture
def capture_mode(monkeypatch, aiter_plugin, sglang_hooks):
    """register() under AKLR_AITER_PLUGIN=capture, aiter having enabled AKLR and another provider before."""
    monkeypatch.setenv(ENV_AITER_PLUGIN, "capture")
    aiter_plugin.configure(source="framework", enable=["aklr", "other"])
    plugin.register()
    ((around, kind),) = sglang_hooks[CAPTURE]
    assert kind == "around"
    return SimpleNamespace(aiter=aiter_plugin, around=around)


def test_outside_a_capture_aiter_does_not_ask_aklr(capture_mode):
    assert capture_mode.aiter.status()["enabled"] == ("other",)


def test_a_decode_graph_capture_asks_aklr_first(capture_mode):
    enabled = []

    def capture_one_shape(self, size, forward):
        enabled.append(capture_mode.aiter.status()["enabled"])
        return f"graph of {size}"

    assert capture_mode.around(capture_one_shape, "runner", 8, forward=None) == "graph of 8"
    assert enabled == [("aklr", "other")]
    assert capture_mode.aiter.status()["enabled"] == ("other",)


def test_a_failed_capture_leaves_aklr_out_again(capture_mode):
    def capture_one_shape(self, size, forward):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        capture_mode.around(capture_one_shape, "runner", 8, None)
    assert capture_mode.aiter.status()["enabled"] == ("other",)


def test_1_keeps_aklr_asked_for_every_call(monkeypatch, aiter_plugin, sglang_hooks):
    monkeypatch.setenv(ENV_AITER_PLUGIN, "1")
    aiter_plugin.configure(source="framework", enable=["other"])
    plugin.register()
    assert aiter_plugin.status()["enabled"] == ("aklr", "other")
    assert CAPTURE not in sglang_hooks
