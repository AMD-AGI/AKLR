# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@vllm_route``: how a vLLM call into aiter reaches AKLR ops, declared beside them.

vLLM compiles a model's forward with torch.compile, so Python there runs once, while tracing.
A route therefore replaces no traced function: it replaces the aiter function a vLLM function
looks up when it runs, inside one of vLLM's opaque custom ops (``vllm::moe_forward``,
``vllm::rocm_unquantized_gemm``). Their bodies run at
every eager call and at every CUDA graph capture, where the route's choice is recorded."""

from __future__ import annotations

import inspect
import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason

__all__ = ["install", "vllm_route", "vllm_version"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, tuple[str, ...], tuple[str, ...], Callable]] = []


def vllm_route(caller: str, *callees: str, versions: tuple[str, ...]) -> Callable[[Callable], Callable]:
    """Declare that vLLM's ``caller`` reaches AKLR ops through this function where it calls
    ``callees`` (aiter functions, each as the dotted name ``caller`` looks up at run time), on
    the vLLM releases ``versions`` it was tested on; on any other, vLLM runs aiter."""

    def declare(route: Callable) -> Callable:
        _ROUTES.append((caller, callees, versions, route))
        return route

    return declare


def vllm_version() -> str:
    """The installed vLLM's release, without a local label (0.33.0rc1.dev910+g4fab5a074 is
    0.33.0rc1.dev910)."""
    return Version(version("vllm")).public


def install() -> None:
    """Wrap the callees of the routes tested on the installed vLLM."""
    installed = vllm_version()
    hooked, untested = [], []
    for caller, callees, versions, route in _ROUTES:
        if installed not in versions:
            untested.append(caller)
            continue
        source = inspect.getsource(pkgutil.resolve_name(caller))
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
            hooked.append(callee)
    # a warning, so that the line shows at any vLLM log level
    _LOGGER.warning(
        "aklr hooks %d aiter calls of vLLM %s%s",
        len(hooked),
        installed,
        f"; not those of {', '.join(untested)}, untested on it" if untested else "",
    )


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
