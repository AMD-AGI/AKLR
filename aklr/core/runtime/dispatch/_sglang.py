###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch._dispatcher import KernelDispatcher

__all__ = ["SGLANG_HOOKED_METHOD", "register_sglang_plugin"]

SGLANG_HOOKED_METHOD = "sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed"


def register_sglang_plugin() -> None:
    """The ``sglang.srt.plugins`` entry point: starts the dispatcher after ``SGLANG_HOOKED_METHOD``."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        SGLANG_HOOKED_METHOD,
        lambda result, *args, **kwargs: KernelDispatcher.get_instance(),
        HookType.AFTER,
    )
