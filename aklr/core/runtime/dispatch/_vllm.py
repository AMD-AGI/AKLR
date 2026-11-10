# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch._registry import DispatcherRegistry
from aklr.core.runtime.hook_registry import KEEP_RESULT, HookRegistry, HookSpec, HookType

__all__ = ["VLLM_HOOKED_METHOD", "register_vllm_plugin"]

# the worker's device is bound when it returns; vLLM loads plugins before it runs
VLLM_HOOKED_METHOD = "vllm.v1.worker.gpu_worker.Worker.init_device"


def _init_runtime(result, *args, **kwargs):
    DispatcherRegistry.default().init_once()
    return KEEP_RESULT


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
    HookRegistry.default().register(
        HookSpec(
            hook_id="vllm.runtime.init",
            target=VLLM_HOOKED_METHOD,
            hook=_init_runtime,
            hook_type=HookType.AFTER,
            phase="bootstrap",
            order=0,
            source=__name__,
        )
    )
