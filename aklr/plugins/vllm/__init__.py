# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for vLLM, loaded through the ``vllm.general_plugins`` entry point."""

from __future__ import annotations

import functools
import importlib

from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import register_vllm_plugin
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
    register_vllm_plugin()
    env = RuntimeEnv()
    if env.is_all_disabled or not env.is_vllm_plugin_enabled:
        return
    from vllm.v1.worker.gpu_worker import Worker

    # only the workers run aiter: the routes go in after AKLR starts there
    init_device = Worker.init_device

    @functools.wraps(init_device)
    def wrapper(self, *args, **kwargs):
        result = init_device(self, *args, **kwargs)
        _install_once()
        return result

    Worker.init_device = wrapper

    compile_or_warm_up_model = Worker.compile_or_warm_up_model

    @functools.wraps(compile_or_warm_up_model)
    def warm_up(self, *args, **kwargs):
        result = compile_or_warm_up_model(self, *args, **kwargs)
        from aklr.plugins.vllm.warmup import warm_up

        warm_up(self)
        return result

    Worker.compile_or_warm_up_model = warm_up

    # the routes' frames move CPython's frame-chunk boundary onto aiter's hot launch calls
    execute_model = Worker.execute_model

    @functools.wraps(execute_model)
    def execute(self, *args, **kwargs):
        return roomy(execute_model, self, *args, **kwargs)

    Worker.execute_model = execute


@functools.cache
def _install_once() -> None:
    # imports the modules that declare the routes: the layers', the models' and the plugin ops' api
    importlib.import_module(f"{__name__}.layers")
    importlib.import_module(f"{__name__}.models")
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    hook.install()
