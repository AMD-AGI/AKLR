# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

from aklr.core.runtime.dispatch._registry import DispatcherRegistry

__all__ = ["VLLM_HOOKED_METHOD", "register_vllm_plugin"]

# the worker's device is bound when it returns; vLLM loads plugins before it runs
VLLM_HOOKED_METHOD = "vllm.v1.worker.gpu_worker.Worker.init_device"


def register_vllm_plugin() -> None:
    """The ``vllm.general_plugins`` entry point (``setup.py``): wraps ``VLLM_HOOKED_METHOD``
    and, unless AKLR_DISABLE=1, proxies aiter's entry points."""
    from vllm.v1.worker.gpu_worker import Worker

    from aklr.core.platform.env import RuntimeEnv
    from aklr.ops import load_routes

    if not RuntimeEnv().is_all_disabled:
        load_routes()
    init_device = Worker.init_device

    @functools.wraps(init_device)
    def wrapper(self, *args, **kwargs):
        result = init_device(self, *args, **kwargs)
        DispatcherRegistry.default().init_once()
        return result

    Worker.init_device = wrapper
