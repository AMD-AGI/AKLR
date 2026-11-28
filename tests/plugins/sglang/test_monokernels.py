# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""register_specs(): the specs tested on the installed SGLang, unless SGLang loads sgl-amd-mono, which
registers the same backend."""

import importlib
import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import pytest

from aklr.plugins.sglang import hook, monokernels

_SPEC_MODULE = "sglang.kernels.plugin_kernels.amd.spec"
_RESOLUTION_HOOKS_MODULE = "sglang.srt.arg_groups.resolution_hooks"
_OVERRIDES_MODULE = "sglang.srt.arg_groups.overrides"


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
    "names, allowed, registers",
    [
        ((), "", True),
        (("sgl_amd_mono",), "", False),
        (("sgl_amd_mono",), "aklr,sgl_amd_mono", False),
        (("sgl_amd_mono",), "aklr", True),
    ],
    ids=["not installed", "installed", "installed and listed", "installed but not listed"],
)
def test_registers_the_specs_unless_sglang_loads_sgl_amd_mono_too(
    monkeypatch, installed, registered, specs, resolution_steps, names, allowed, registers
):
    """SGLang loads sgl-amd-mono where it is installed and SGLANG_PLUGINS, if set, lists it."""
    installed("aklr", *names)
    monkeypatch.setenv("SGLANG_PLUGINS", allowed)
    specs("0.5.23")
    monokernels.register_specs()
    assert bool(registered) is registers


@pytest.fixture
def specs(monkeypatch):
    """Two specs declared, one tested on 0.5.23 and 0.5.24 and one on 0.5.22; sets the installed release."""
    declared = {
        "Newer": ("package.newer:SPEC", ("0.5.23", "0.5.24")),
        "Older": ("package.older:SPEC", ("0.5.22",)),
    }
    monkeypatch.setattr(monokernels, "SPECS", declared)
    return lambda release: monkeypatch.setattr(hook, "sglang_version", lambda: release)


def test_a_release_no_spec_was_tested_on_registers_none_and_needs_no_spec_module(
    monkeypatch, installed, specs
):
    """A release before the monokernel interface has no spec module: the plugin starts on it all the same."""
    installed("aklr")
    specs("0.5.21")
    # importing it raises
    monkeypatch.setitem(sys.modules, _SPEC_MODULE, None)
    monokernels.register_specs()


def test_registers_the_specs_tested_on_the_installed_release_and_logs_the_others(
    installed, registered, specs, caplog
):
    installed("aklr")
    specs("0.5.23")
    monokernels.register_specs()
    assert registered == [{"backend": "amd", "architecture": "Newer", "path": "package.newer:SPEC"}]
    assert "Older" in caplog.text and "Newer" not in caplog.text and "0.5.23" in caplog.text


@pytest.fixture
def resolution_steps(monkeypatch):
    """SGLang's server-argument resolution: the steps plugins wrap, by name, on server arguments that name
    the model's architectures."""
    steps = {}
    hooks = ModuleType(_RESOLUTION_HOOKS_MODULE)
    hooks.register_resolution_hook = lambda name: lambda step: steps.setdefault(name, step)
    overrides = ModuleType(_OVERRIDES_MODULE)
    overrides.resolving_view = lambda server_args: server_args
    overrides.model_config_of = lambda server_args: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=server_args.architectures)
    )
    monkeypatch.setitem(sys.modules, _RESOLUTION_HOOKS_MODULE, hooks)
    monkeypatch.setitem(sys.modules, _OVERRIDES_MODULE, overrides)
    return steps


def test_beside_sgl_amd_mono_the_flag_on_a_model_it_has_a_spec_for_asks_for_one_plugin(
    monkeypatch, installed, registered, specs, resolution_steps
):
    """SGLang takes one spec an architecture: beside sgl-amd-mono, aklr registers none, and
    --enable-amd-monokernel on a model it has a spec for stops start-up, naming both plugins and how to
    choose one."""
    installed("aklr", "sgl_amd_mono")
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    specs("0.5.23")
    monokernels.register_specs()
    assert registered == []
    ran = []
    with pytest.raises(ValueError, match="aklr.*Newer.*sgl_amd_mono.*SGLANG_PLUGINS"):
        resolution_steps["handle_amd_monokernel"](
            SimpleNamespace(enable_amd_monokernel=True, architectures=["Newer"]), ran.append
        )
    assert ran == []


@pytest.mark.parametrize(
    "flag, architectures",
    [(False, ["Newer"]), (True, ["Other"]), (True, ["Older"])],
    ids=["no flag", "a model aklr has no spec for", "a spec untested on the release"],
)
def test_beside_sgl_amd_mono_any_other_start_up_runs_sglang_s_own_step(
    monkeypatch, installed, registered, specs, resolution_steps, flag, architectures
):
    """Only the flag on a model aklr serves asks for one plugin: sgl-amd-mono serves the others."""
    installed("aklr", "sgl_amd_mono")
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    specs("0.5.23")
    monokernels.register_specs()
    server_args = SimpleNamespace(enable_amd_monokernel=flag, architectures=architectures)
    ran = []
    resolution_steps["handle_amd_monokernel"](server_args, ran.append)
    assert ran == [server_args]


def test_every_spec_was_tested_on_the_installed_sglang():
    """A new sglang image needs the specs tested on it."""
    try:
        installed = hook.sglang_version()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang is not installed")
    assert [arch for arch, (_, versions) in monokernels.SPECS.items() if installed not in versions] == []


def test_the_flag_s_resolution_step_is_overridable_on_the_installed_sglang():
    """Beside sgl-amd-mono, aklr wraps SGLang's --enable-amd-monokernel step by its name, with the server
    arguments read as that step reads them: a new sglang image needs them as they are there."""
    hooks = pytest.importorskip(_RESOLUTION_HOOKS_MODULE)
    overrides = pytest.importorskip(_OVERRIDES_MODULE)
    # SGLang checks the name before it takes a wrapper, so naming the step wraps nothing
    hooks.register_resolution_hook("handle_amd_monokernel")
    assert callable(overrides.model_config_of) and callable(overrides.resolving_view)


def test_every_spec_is_a_monospec_of_the_installed_api():
    """Each spec imports on the installed SGLang, written against its MONOKERNEL_API_VERSION."""
    amd = pytest.importorskip(_SPEC_MODULE)
    for path, _ in monokernels.SPECS.values():
        module, _, attr = path.partition(":")
        spec = getattr(importlib.import_module(module), attr)
        assert isinstance(spec, amd.MonoSpec) and spec.api_version == amd.MONOKERNEL_API_VERSION, path
