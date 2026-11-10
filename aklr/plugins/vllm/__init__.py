# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for vLLM, loaded through the ``vllm.general_plugins`` entry point."""

from __future__ import annotations

import functools
import importlib

from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime import _hook_registry as _hooks
from aklr.core.runtime.dispatch._vllm import _register_runtime_hook
from aklr.plugins.vllm import hook
from aklr.plugins.vllm.stack import roomy

__all__ = ["register"]


def register() -> None:
    """Entry point vLLM calls in each of its processes; a plugin that cannot start ends the
    process."""
    try:
        _register()
    except Exception as e:
        raise SystemExit(f"aklr: the vllm plugin cannot start: {e!r}") from e


def _register() -> None:
    # Validate switches before changing any framework bindings.
    env = RuntimeEnv()
    enabled = not env.is_all_disabled and env.is_vllm_plugin_enabled
    _register_runtime_hook()
    registry = _hooks.HookRegistry.default()
    if enabled:
        for name, method, callback, hook_type, order in (
            ("routes", "init_device", _install_routes, _hooks.HookType.AFTER, 10),
            ("warmup", "compile_or_warm_up_model", _warm_up, _hooks.HookType.AFTER, 0),
            ("execute", "execute_model", _execute, _hooks.HookType.AROUND, 0),
        ):
            registry.register(
                _hooks.HookSpec(
                    hook_id=f"vllm.plugin.{name}",
                    target=f"vllm.v1.worker.gpu_worker.Worker.{method}",
                    hook=callback,
                    hook_type=hook_type,
                    phase="bootstrap",
                    order=order,
                    source=__name__,
                )
            )
    registry.apply("bootstrap")


def _install_routes(result, *args, **kwargs):
    # The runtime AFTER hook is inside this one, so initialization has completed.
    _install_once()
    return _hooks.KEEP_RESULT


def _warm_up(result, worker, *args, **kwargs):
    # Preserve the existing supplementary warmup and its version-specific runner calls.
    if _install_once():
        from aklr.plugins.vllm.warmup import warm_up

        warm_up(worker)
    return _hooks.KEEP_RESULT


def _execute(next_fn, *args, **kwargs):
    # Keep AITER's hot launches away from CPython's frame-chunk boundary.
    return roomy(next_fn, *args, **kwargs)


@functools.cache
def _install_once() -> list[str]:
    # imports the modules that declare the routes: the layers', the models' and the plugin ops' api
    importlib.import_module(f"{__name__}.layers")
    importlib.import_module(f"{__name__}.models")
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    return hook._install()
