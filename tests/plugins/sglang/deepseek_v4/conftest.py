# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stand-ins for the few sglang modules the plugin imports, so its tests need neither sglang nor a GPU."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.plugins.sglang.deepseek_v4 import _monokernel_spec


@pytest.fixture
def sglang(monkeypatch):
    """A recording HookRegistry, resolution-hook registry and monokernel spec registry; the
    monokernel backend (``state.backend``), its startup check (``state.disable_reasons``)
    and the resolving view (``state.view``) are the tests' to set."""
    state = SimpleNamespace(
        registered=[],
        specs=[],
        resolution_hooks={},
        declared=[],
        checked=[],
        backend=None,
        disable_reasons=[],
        view=SimpleNamespace(monokernel_backend=None),
        architectures=["DeepseekV4ForCausalLM"],
    )

    def register_resolution_hook(name):
        return lambda fn: state.resolution_hooks.setdefault(name, fn)

    def disable_reasons(*, cfg, model_config, isa):
        state.checked.append(cfg.monokernel_backend)
        return None, list(state.disable_reasons)

    def shape_key(forward_batch):
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return None
        return ("target_verify", forward_batch.batch_size)

    modules = {
        "sglang.srt.plugins.hook_registry": {
            "HookType": SimpleNamespace(AROUND="around"),
            "HookRegistry": SimpleNamespace(register=lambda *args: state.registered.append(args)),
        },
        "sglang.srt.arg_groups.resolution_hooks": {"register_resolution_hook": register_resolution_hook},
        "sglang.srt.arg_groups.overrides": {
            "declare_resolution": lambda server_args, source, **fields: state.declared.append(
                (source, fields)
            ),
            "model_config_of": lambda server_args: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=state.architectures)
            ),
            "resolving_view": lambda server_args: state.view,
        },
        "sglang.srt.monokernels.spec": {
            **{n: getattr(_monokernel_spec, n) for n in _monokernel_spec.__dict__ if n[:1].isupper()},
            "register": lambda **kwargs: state.specs.append(kwargs),
        },
        "sglang.srt.monokernels.guards": {
            "device_isa": lambda: "gfx950",
            "monokernel_disable_reasons": disable_reasons,
        },
        "sglang.srt.monokernels.runtime": {
            "_backend": lambda: state.backend,
            "shape_key": shape_key,
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture(autouse=True)
def no_leaked_modules():
    """Drop the plugin modules a test imported: one imported over the stand-ins above
    would serve the next test, which may see the installed sglang."""
    package = "aklr.plugins.sglang.models.deepseek_v4"
    before = set(sys.modules)
    yield
    for name in [n for n in sys.modules if n.startswith(f"{package}.") and n not in before]:
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            sys.modules[parent].__dict__.pop(child, None)


@pytest.fixture(autouse=True)
def serving(monkeypatch):
    """A process that built a DeepSeek-V4.1 target on gfx950 at TP 2 or 4, as the routes'
    tests assume; a test of another setup sets ``_SERVING`` back to False."""
    import aklr.plugins.sglang.models.deepseek_v4 as dsv4

    monkeypatch.setattr(dsv4, "_SERVING", True)
