# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.fastkey._alloc_outputs_fn import AllocOutputsFnGenerator
from aklr.core.contract.fastkey._codec import CallArgs, FastKey, FastKeyCodec
from aklr.core.contract.fastkey._fast_key_fn import FastKeyFnGenerator

__all__ = [
    "AllocOutputsFnGenerator",
    "CallArgs",
    "FastKey",
    "FastKeyCodec",
    "FastKeyFnGenerator",
]
