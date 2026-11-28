# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def aiter_plugin(monkeypatch):
    """aiter's plugin interface with AKLR and one other provider installed."""
    plugin = pytest.importorskip("aiter.plugin")
    provider = importlib.import_module("aklr.plugins.aiter.provider")
    plugin._reset_for_testing()
    other = SimpleNamespace(
        name="other", api_versions={plugin.API_VERSION}, op_versions={}, select=lambda *a: None
    )
    entry_points = tuple(
        SimpleNamespace(name=target.name, load=lambda target=target: target)
        for target in (provider.provider, other)
    )
    monkeypatch.setattr(plugin, "_entry_points", lambda: entry_points)
    monkeypatch.setattr(provider, "ready", lambda: True)
    yield plugin
    plugin._reset_for_testing()
