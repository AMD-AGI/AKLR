# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch._registry import DispatcherRegistry

__all__ = ["SGLANG_HOOKED_METHOD", "register_sglang_plugin"]

SGLANG_HOOKED_METHOD = "sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed"


def register_sglang_plugin() -> None:
    """Start AKLR after ``SGLANG_HOOKED_METHOD``; the sglang plugin's entry point calls it."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        SGLANG_HOOKED_METHOD,
        lambda result, *args, **kwargs: DispatcherRegistry.default().init_once(),
        HookType.AFTER,
    )
