# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime import _vllm_hook_registry as _hooks
from aklr.core.runtime.dispatch._registry import DispatcherRegistry

__all__ = ["VLLM_HOOKED_METHOD", "register_vllm_plugin"]

# the worker's device is bound when it returns; vLLM loads plugins before it runs
VLLM_HOOKED_METHOD = "vllm.v1.worker.gpu_worker.Worker.init_device"


def _init_runtime(result, *args, **kwargs):
    DispatcherRegistry.default().init_once()
    return _hooks.KEEP_RESULT


def register_vllm_plugin() -> None:
    """Start AKLR after ``VLLM_HOOKED_METHOD``, including enabled vLLM routes.

    Both public entry points use the plugin's bootstrap coordinator so the first
    call collects every enabled hook before freezing the phase. The local import
    keeps importing the core runtime independent of the optional plugin modules.
    """
    from aklr.plugins.vllm import _register

    _register()


def _register_runtime_hook() -> None:
    """Declare core initialization; the plugin owns the complete bootstrap batch."""
    _hooks.HookRegistry.default().register(
        _hooks.HookSpec(
            hook_id="vllm.runtime.init",
            target=VLLM_HOOKED_METHOD,
            hook=_init_runtime,
            hook_type=_hooks.HookType.AFTER,
            phase="bootstrap",
            order=0,
            source=__name__,
        )
    )
