# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Calls a framework function makes into another library (aiter), routed to AKLR ops: each
callee is wrapped in its own module, where the caller looks it up when it runs. The call routes
of the vLLM and sglang plugins."""

from __future__ import annotations

import inspect
import pkgutil
from collections.abc import Callable, Iterable

import torch

from aklr.core.runtime.dispatch import FallbackReason

__all__ = ["wrap_calls"]


def wrap_calls(caller: str, callees: Iterable[str], route: Callable) -> list[str]:
    """Wrap each of ``callees`` (the dotted name ``caller`` looks up when it runs) so that
    ``route`` takes the call first; return them. Raises if ``caller`` does not call one;
    wrapping again with the same ``route`` changes nothing."""
    source = inspect.getsource(pkgutil.resolve_name(caller))
    wrapped = []
    for callee in callees:
        owner_name, attr = callee.rsplit(".", 1)
        if f"{attr}(" not in source:
            raise RuntimeError(f"{caller} does not call {callee}")
        pkgutil.resolve_name(owner_name)
        # aiter's backward-compatible module paths change sys.modules on their first import
        owner = pkgutil.resolve_name(owner_name)
        orig = getattr(owner, attr)
        if getattr(orig, "aklr_route", None) is not route:
            setattr(owner, attr, _around(route, orig))
        wrapped.append(callee)
    return wrapped


def _around(route: Callable, orig: Callable) -> Callable:
    def around(*args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            result = route(*args, **kwargs)
            if not isinstance(result, FallbackReason):
                return result
        return orig(*args, **kwargs)

    around.__wrapped__ = orig
    around.aklr_route = route
    return around
