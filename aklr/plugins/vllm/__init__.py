# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for vLLM, loaded through the ``vllm.general_plugins`` entry point."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime import _vllm_hook_registry as _hooks
from aklr.core.runtime.dispatch._vllm import _register_runtime_hook
from aklr.plugins.vllm import hook
from aklr.plugins.vllm.stack import roomy

__all__ = ["AFTER_STEP", "register"]

# Called after each step a worker executes (execute_model: its forward is launched), on every rank alike: an op's
# watch over state its kernels keep between steps (the latent-MoE tail's fail-stop), registered when the op sets that
# state up. A watch may return a check of the step, which runs before the step's output leaves the worker: as an
# async output is read (after vLLM's own wait for it), or before a synchronous one is returned; raising there keeps
# the output from the scheduler.
AFTER_STEP: list[Callable[[], Callable[[], None] | None]] = []
# the checks of the step last executed, until its output is sampled
_STEP_CHECKS: list[Callable[[], None]] = []
_SAMPLE_TOKENS = "vllm.v1.worker.gpu_worker.Worker.sample_tokens"


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
    if env.is_aiter_plugin_enabled and not env.is_all_disabled:
        from aklr.plugins.aiter.provider import enable

        enable("AKLR vllm plugin, AKLR_AITER_PLUGIN=1")
    enabled = not env.is_all_disabled and env.is_vllm_plugin_enabled
    _register_runtime_hook()
    registry = _hooks.HookRegistry.default()
    if enabled:
        hooks = [
            ("routes", "init_device", _install_routes, _hooks.HookType.AFTER, 10),
            ("warmup", "compile_or_warm_up_model", _warm_up, _hooks.HookType.AFTER, 0),
            ("execute", "execute_model", _execute, _hooks.HookType.AROUND, 0),
        ]
        if _exists(_SAMPLE_TOKENS):  # a vLLM that samples apart from execute_model
            hooks.append(("sample", "sample_tokens", _sample, _hooks.HookType.AROUND, 0))
        for name, method, callback, hook_type, order in hooks:
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


def _exists(path: str) -> bool:
    try:
        pkgutil.resolve_name(path)
    except (ImportError, AttributeError):
        return False
    return True


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
    result = roomy(next_fn, *args, **kwargs)
    checks = [check for check in (after_step() for after_step in AFTER_STEP) if check is not None]
    # None: the step's output comes from sample_tokens
    _STEP_CHECKS[:] = checks if result is None else []
    return _checked(result, checks)


def _sample(next_fn, *args, **kwargs):
    checks, _STEP_CHECKS[:] = list(_STEP_CHECKS), []
    return _checked(next_fn(*args, **kwargs), checks)


def _checked(output: Any, checks: list[Callable[[], None]]) -> Any:
    """``output`` (vLLM's model runner output, None, or an async one to read later) leaving the worker only through
    ``checks``: an async one runs them as it is read, before its own read; any other runs them now."""
    if output is None or not checks:
        return output
    read = getattr(output, "get_output", None)
    if read is None:
        for check in checks:
            check()
        return output

    def get_output():
        for check in checks:
            check()
        return read()

    output.get_output = get_output
    return output


def _install_once() -> list[str]:
    """Import route declarations and let the registry install them idempotently."""
    # The registry owns installation state, including restore/retry. Do not cache
    # an installed target list here: restore invalidates it. Python caches imports.
    importlib.import_module(f"{__name__}.layers")
    importlib.import_module(f"{__name__}.models")
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    return hook._install()
