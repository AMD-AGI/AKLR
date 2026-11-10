# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from sglang to AKLR ops, declared beside the ops.

``@sglang_route`` replaces a function that sglang defines. ``@sglang_call_route`` replaces a
function that sglang calls but does not define, such as aiter's ``fused_moe``; it is replaced in
its own module, where sglang looks it up at call time."""

from __future__ import annotations

import inspect
import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.runtime.dispatch._trace import trace

__all__ = ["install", "sglang_call_route", "sglang_route", "sglang_version"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, tuple[str, ...], Callable]] = []
_CALL_ROUTES: list[tuple[str, tuple[str, ...], tuple[str, ...], Callable]] = []


def sglang_route(target: str, *, versions: tuple[str, ...]) -> Callable[[Callable], Callable]:
    """Declare that sglang's ``target`` reaches AKLR ops through this function, on the sglang
    releases ``versions`` it was tested on; on any other, sglang runs its own code."""

    def declare(route: Callable) -> Callable:
        _ROUTES.append((target, versions, route))
        return route

    return declare


def sglang_call_route(
    caller: str, *callees: str, versions: tuple[str, ...]
) -> Callable[[Callable], Callable]:
    """Declare this function as the route for the calls that sglang's ``caller`` makes to
    ``callees``, each the dotted name that ``caller`` looks up when it runs. The route is
    installed only on the sglang releases in ``versions``, the ones it was tested on."""

    def declare(route: Callable) -> Callable:
        _CALL_ROUTES.append((caller, callees, versions, route))
        return route

    return declare


def sglang_version() -> str:
    """The installed sglang's release, without a local label (0.5.22+rocm is 0.5.22)."""
    return Version(version("sglang")).public


def install() -> None:
    """Wrap the targets of the routes tested on the installed sglang."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    installed = sglang_version()
    functions, calls, untested = 0, [], []
    for target, versions, route in _ROUTES:
        if installed in versions:
            # sglang only logs a target it cannot resolve
            pkgutil.resolve_name(target)
            HookRegistry.register(target, _around(route, target), HookType.AROUND)
            functions += 1
        else:
            untested.append(target)
    for caller, callees, versions, route in _CALL_ROUTES:
        if installed not in versions:
            untested.append(caller)
            continue
        source = inspect.getsource(pkgutil.resolve_name(caller))
        for callee in callees:
            if f"{callee.rsplit('.', 1)[1]}(" not in source:
                raise RuntimeError(f"{caller} does not call {callee}")
            _wrap_call(callee, route)
            calls.append(callee)
    # a warning, so that the line shows at any sglang log level
    _LOGGER.warning(
        "aklr hooks %d sglang functions and %d calls on sglang %s%s",
        functions,
        len(calls),
        installed,
        f"; not {', '.join(untested)}, untested on it" if untested else "",
    )


def _around(route: Callable, target: str) -> Callable:
    route_name = f"{route.__module__}.{route.__qualname__}"

    def around(orig, *args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            result = route(*args, **kwargs)
            fallback = result if isinstance(result, FallbackReason) else None
            if (tracer := trace()) is not None and tracer.is_plugin:
                tracer.plugin("sglang", target, route_name, fallback)
            if fallback is None:
                return result
        return orig(*args, **kwargs)

    return around


def _wrap_call(callee: str, route: Callable) -> None:
    """Replace ``callee`` in its module with a wrapper that tries ``route`` first, as ``_around``
    does for a function route. Wrapping again with the same route changes nothing."""
    owner_name, attr = callee.rsplit(".", 1)
    # Resolve twice: on its first import, aiter replaces some of its module paths in sys.modules,
    # so the first lookup can return a module that later imports no longer return.
    pkgutil.resolve_name(owner_name)
    owner = pkgutil.resolve_name(owner_name)
    orig = getattr(owner, attr)
    if getattr(orig, "aklr_route", None) is route:
        return
    around = _around(route, callee)

    def call(*args, **kwargs):
        return around(orig, *args, **kwargs)

    call.__wrapped__ = orig
    call.aklr_route = route
    setattr(owner, attr, call)
