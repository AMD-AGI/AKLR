# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every framework's adapters, declared to the seam registry on import."""

from __future__ import annotations

from collections.abc import Callable

from aklr.core.runtime.seam import Adapter, SeamRegistry

__all__ = ["adapter"]


def adapter(entry: str, api: Callable, baseline: Callable | None = None) -> Callable[[Callable], Callable]:
    """Declare ``entry`` served through the AKLR api ``api``: the decorated function
    takes the entry point's parameters and returns the api's kwargs, or None to
    decline; ``baseline`` takes the entry point's function and an api call's kwargs
    and returns a call of the entry point standing for it, or None to decline."""

    def register(fn: Callable) -> Callable:
        SeamRegistry.default().add(Adapter(entry, api, fn, baseline))
        return fn

    return register


# Last: the framework modules import ``adapter`` from here.
from aklr.seam import aiter, sglang  # noqa: F401
