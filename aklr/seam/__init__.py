###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from collections.abc import Callable

from aklr.core.runtime.seam import Adapter, SeamRegistry

__all__ = ["adapter"]


def adapter(entry: str, api: Callable, baseline: Callable | None = None) -> Callable[[Callable], Callable]:
    """Serve entry by api: the decorated fn maps entry's arguments to api kwargs, or None to decline."""

    def register(fn: Callable) -> Callable:
        SeamRegistry.default().add(Adapter(entry, api, fn, baseline))
        return fn

    return register


# last: aiter and sglang import adapter from this module
from aklr.seam import aiter, sglang  # noqa: F401
