# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""register_specs(): every spec, unless SGLang loads sgl-amd-mono, which registers the same backend."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from aklr.plugins.sglang import monokernels

_SPEC_MODULE = "sglang.kernels.plugin_kernels.amd.spec"


@pytest.fixture
def installed(monkeypatch):
    """Sets the installed sglang.srt.plugins entry points by name."""

    def install(*names):
        eps = [SimpleNamespace(name=n) for n in names]
        monkeypatch.setattr(monokernels, "entry_points", lambda group: eps)

    return install


@pytest.fixture
def registered(monkeypatch):
    """SGLang's register(), recording its calls."""
    calls = []
    module = ModuleType(_SPEC_MODULE)
    module.register = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, _SPEC_MODULE, module)
    return calls


@pytest.mark.parametrize(
    "names, allowed, loads",
    [
        ((), "", False),
        (("sgl_amd_mono",), "", True),
        (("sgl_amd_mono",), "aklr,sgl_amd_mono", True),
        (("sgl_amd_mono",), "aklr", False),
    ],
)
def test_sgl_amd_mono_loads_when_installed_and_allowed(monkeypatch, installed, names, allowed, loads):
    installed(*names)
    monkeypatch.setenv("SGLANG_PLUGINS", allowed)
    assert monokernels._loads_sgl_amd_mono() is loads


def test_registers_every_spec(installed, registered):
    installed("aklr")
    monokernels.register_specs()
    assert registered == [
        {"backend": "amd", "architecture": arch, "path": path} for arch, path in monokernels.SPECS.items()
    ]


def test_leaves_the_backend_to_sgl_amd_mono(monkeypatch, installed, registered):
    installed("aklr", "sgl_amd_mono")
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    monokernels.register_specs()
    assert registered == []


def test_every_spec_is_a_monospec_of_the_installed_api():
    """Each spec imports on the installed SGLang, written against its MONOKERNEL_API_VERSION."""
    amd = pytest.importorskip(_SPEC_MODULE)
    for path in monokernels.SPECS.values():
        module, _, attr = path.partition(":")
        spec = getattr(importlib.import_module(module), attr)
        assert isinstance(spec, amd.MonoSpec) and spec.api_version == amd.MONOKERNEL_API_VERSION, path
