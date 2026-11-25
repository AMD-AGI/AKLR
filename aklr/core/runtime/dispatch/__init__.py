###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch._config import DispatchConfig, DispatchConfigRegistry
from aklr.core.runtime.dispatch._dispatch_key import DispatchKey
from aklr.core.runtime.dispatch._dispatcher import Dispatcher
from aklr.core.runtime.dispatch._kernel_function import KernelFunction
from aklr.core.runtime.dispatch._kernel_tuner import KernelTuner
from aklr.core.runtime.dispatch._operator_entry import OperatorEntry
from aklr.core.runtime.dispatch._ops import OpVariants, aklr_api
from aklr.core.runtime.dispatch._sglang import (
    SGLANG_HOOKED_METHOD,
    register_sglang_plugin,
)
from aklr.core.runtime.dispatch._vllm import VLLM_HOOKED_METHOD, register_vllm_plugin

__all__ = [
    "SGLANG_HOOKED_METHOD",
    "VLLM_HOOKED_METHOD",
    "DispatchConfig",
    "DispatchConfigRegistry",
    "DispatchKey",
    "Dispatcher",
    "KernelFunction",
    "KernelTuner",
    "OpVariants",
    "OperatorEntry",
    "aklr_api",
    "register_sglang_plugin",
    "register_vllm_plugin",
]
