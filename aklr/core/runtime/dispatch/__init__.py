# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.runtime.dispatch._dispatcher import (
    DispatchTable,
    FallbackReason,
    KernelDispatcher,
)
from aklr.core.runtime.dispatch._recorder import CallRecorder
from aklr.core.runtime.dispatch._registry import DispatcherRegistry, api
from aklr.core.runtime.dispatch._sglang import (
    SGLANG_HOOKED_METHOD,
    register_sglang_plugin,
)
from aklr.core.runtime.dispatch._vllm import VLLM_HOOKED_METHOD, register_vllm_plugin

__all__ = [
    "SGLANG_HOOKED_METHOD",
    "VLLM_HOOKED_METHOD",
    "CallRecorder",
    "DispatchTable",
    "DispatcherRegistry",
    "FallbackReason",
    "KernelDispatcher",
    "api",
    "register_sglang_plugin",
    "register_vllm_plugin",
]
