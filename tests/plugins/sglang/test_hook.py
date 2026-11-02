# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin switch, one route per target for the running sglang, and falling back to sglang."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import aklr.ops
import aklr.plugins.sglang as plugin
from aklr.core.platform.env import ENV_DISABLE, ENV_SGLANG_PLUGIN
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import hook, stats


def original(x):
    return f"sglang({x})"


@pytest.fixture
def logger(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(hook, "_LOGGER", fake)
    return fake


def installed(monkeypatch, sglang, running, *routes):
    """The wrapper sglang gets for one target declaring ``routes``, (versions, route) pairs, on sglang ``running``."""
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.setattr(hook, "_INSTALLED", {})
    monkeypatch.setattr(hook, "version", lambda _: running)
    for versions, route in routes:
        hook.sglang_route("pkg.Layer.forward", versions)(route)
    hook.install()
    (wrapper,) = [w for t, w, kind in sglang.registered if t == "pkg.Layer.forward"]
    return wrapper


@pytest.fixture
def register_calls(monkeypatch):
    """What register() did, in order, its plugin op package being aklr.plugins.sglang.x.y
    and its one model backend aklr.plugins.sglang.models.m."""
    calls = []
    monkeypatch.setattr(plugin, "register_sglang_plugin", lambda: calls.append("aklr starts"))
    monkeypatch.setattr(plugin, "op_packages", lambda: ("aklr.ops.gemm", "aklr.plugins.sglang.x.y"))
    monkeypatch.setattr(plugin, "BACKENDS", ("aklr.plugins.sglang.models.m",))
    monkeypatch.setattr(plugin, "_attach_log_handler", lambda: None)
    monkeypatch.setattr(aklr.ops, "load_routes", lambda strict: calls.append("load_routes") or [])

    def import_module(name):
        calls.append(name)
        return SimpleNamespace(register=lambda: calls.append(f"{name} registers"))

    monkeypatch.setattr(plugin.importlib, "import_module", import_module)
    monkeypatch.setattr(plugin, "install", lambda: calls.append("install"))
    monkeypatch.setattr(
        plugin.peer_memory,
        "set_group",
        lambda group: calls.append(f"peers across {group.__name__}"),
    )
    return calls


@pytest.mark.parametrize(
    "env, proxied, hooked",
    [
        ({}, True, False),
        ({ENV_SGLANG_PLUGIN: "0"}, True, False),
        ({ENV_SGLANG_PLUGIN: "1"}, True, True),
        ({ENV_SGLANG_PLUGIN: "1", ENV_DISABLE: "1"}, False, False),
        ({ENV_SGLANG_PLUGIN: "1", ENV_DISABLE: "aklr.plugins"}, True, False),
    ],
)
def test_only_1_turns_the_routes_on(monkeypatch, register_calls, env, proxied, hooked):
    """aiter's entry points are proxied unless AKLR_DISABLE=1; the layer routes hook and the
    model backends register only with AKLR_SGLANG_PLUGIN=1."""
    monkeypatch.delenv(ENV_DISABLE, raising=False)
    monkeypatch.delenv(ENV_SGLANG_PLUGIN, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    plugin.register()
    routes = (
        [
            "aklr.plugins.sglang.x.y.api",
            "aklr.plugins.sglang.models.m",
            "aklr.plugins.sglang.models.m registers",
            "install",
            "peers across _tp_group",
        ]
        if hooked
        else []
    )
    assert register_calls == ["aklr starts", *(["load_routes"] * proxied), *routes]


def test_any_other_switch_value_is_an_error(monkeypatch, register_calls):
    monkeypatch.setenv(ENV_SGLANG_PLUGIN, "true")
    with pytest.raises(ValueError, match=ENV_SGLANG_PLUGIN):
        plugin.register()


def test_the_route_for_the_running_sglang_serves(monkeypatch, sglang, logger):
    wrapper = installed(
        monkeypatch,
        sglang,
        "0.5.20",
        (">=0.5.18,<0.5.19", lambda x: f"old({x})"),
        (">=0.5.19,<0.5.21", lambda x: f"new({x})"),
    )
    assert wrapper(original, 1) == "new(1)"
    logger.warning.assert_not_called()


def test_a_dev_build_takes_its_release_route(monkeypatch, sglang):
    wrapper = installed(
        monkeypatch,
        sglang,
        "0.5.19.dev20260906+g5bebe7a033",
        (">=0.5.19,<0.5.21", lambda x: f"new({x})"),
    )
    assert wrapper(original, 1) == "new(1)"


def test_a_fallback_reason_runs_sglang(monkeypatch, sglang):
    wrapper = installed(
        monkeypatch,
        sglang,
        "0.5.19",
        (">=0.5.19,<0.5.21", lambda x: FallbackReason.NOT_MEASURED),
    )
    assert wrapper(original, 1) == "sglang(1)"


def test_a_raising_route_raises(monkeypatch, sglang):
    def boom(x):
        raise RuntimeError("boom")

    wrapper = installed(monkeypatch, sglang, "0.5.20", (">=0.5.19,<0.5.21", boom))
    with pytest.raises(RuntimeError, match="boom"):
        wrapper(original, 1)


@pytest.mark.parametrize(
    "ranges",
    [[">=0.5.21,<0.5.22"], [">=0.5.19,<0.5.21", ">=0.5.20,<0.5.21"]],
    ids=["none", "two"],
)
def test_no_single_route_warns_once_and_runs_sglang(monkeypatch, sglang, logger, ranges):
    routes = [(versions, lambda x: "served") for versions in ranges]
    wrapper = installed(monkeypatch, sglang, "0.5.20", *routes)
    assert [wrapper(original, x) for x in (1, 2)] == ["sglang(1)", "sglang(2)"]
    logger.warning.assert_called_once()


def test_a_range_without_an_upper_bound_is_an_error():
    with pytest.raises(ValueError, match="no upper bound"):
        hook.sglang_route("pkg.Layer.forward", ">=0.5.20")


def test_a_route_fallback_runs_sglang_and_counts_its_reason(monkeypatch, sglang, fresh_stats):
    wrapper = installed(
        monkeypatch,
        sglang,
        "0.5.22",
        (">=0.5.22,<0.5.23", lambda x: hook.RouteFallback("extend", f"x={x}")),
    )
    assert [wrapper(original, x) for x in (1, 2)] == ["sglang(1)", "sglang(2)"]
    stats.dump()
    (export,) = fresh_stats.glob("route_calls_*.json")
    (calls,) = json.loads(export.read_text())["calls"]
    assert calls == {
        "target": "pkg.Layer.forward",
        "outcome": "fallback",
        "reason": "extend",
        "detail": "x=1",
        "calls": 2,
    }


def test_a_served_call_is_counted(monkeypatch, sglang):
    wrapper = installed(monkeypatch, sglang, "0.5.22", (">=0.5.22,<0.5.23", lambda x: f"new({x})"))
    wrapper(original, 1)
    assert [(c["outcome"], c["calls"]) for c in stats.snapshot()["calls"]] == [("served", 1)]


def test_installed_routes_name_why_a_target_is_unrouted(monkeypatch, sglang):
    installed(monkeypatch, sglang, "0.5.20", (">=0.5.21,<0.5.22", lambda x: "served"))
    assert hook.installed_routes()["pkg.Layer.forward"].startswith("0 routes")


def test_an_around_route_runs_around_sglang_and_counts_nothing(monkeypatch, sglang):
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.setattr(hook, "_INSTALLED", {})
    monkeypatch.setattr(hook, "version", lambda _: "0.5.22")
    seen = []

    def step(orig, x):
        seen.append("before")
        out = orig(x)
        seen.append("after")
        return out

    hook.sglang_route("pkg.Runner.load", ">=0.5.22,<0.5.23", around=True)(step)
    hook.install()
    (wrapper,) = [w for t, w, kind in sglang.registered if t == "pkg.Runner.load"]
    assert wrapper(original, 3) == "sglang(3)" and seen == ["before", "after"]
    assert stats.snapshot()["calls"] == []
