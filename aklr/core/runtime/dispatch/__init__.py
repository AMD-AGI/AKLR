###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union, overload

from aklr.core.runtime.dispatch._aklr_api import aklr_api
from aklr.core.runtime.dispatch._dispatch_table import DispatchTable
from aklr.core.runtime.dispatch._dispatcher import (
    DispatchConfig,
    DispatchConfigRegistry,
    KernelDispatcher,
)
from aklr.core.runtime.dispatch._kernel_tuner import KernelTuner
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
    "DispatchTable",
    "KernelDispatcher",
    "KernelTuner",
    "register_sglang_plugin",
    "register_vllm_plugin",
    "aklr_api",
]


logger = logging.getLogger(__name__)
