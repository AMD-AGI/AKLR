###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools

from aklr.core.runtime.dispatch._dispatcher import KernelDispatcher

__all__ = ["VLLM_HOOKED_METHOD", "register_vllm_plugin"]

# the worker's device is bound once it returns; vLLM loads plugins before it runs
VLLM_HOOKED_METHOD = "vllm.v1.worker.gpu_worker.Worker.init_device"


def register_vllm_plugin() -> None:
    """The ``vllm.general_plugins`` entry point: starts the dispatcher after ``VLLM_HOOKED_METHOD``."""
    from vllm.v1.worker.gpu_worker import Worker

    init_device = Worker.init_device

    @functools.wraps(init_device)
    def wrapper(self, *args, **kwargs):
        result = init_device(self, *args, **kwargs)
        KernelDispatcher.get_instance()
        return result

    Worker.init_device = wrapper
