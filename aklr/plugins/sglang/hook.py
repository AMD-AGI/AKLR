# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@sglang_route``: how a sglang function reaches AKLR ops, declared beside them.

``@sglang_call_route``: how a call sglang makes reaches AKLR ops, where the callee is not
sglang's (aiter's ``fused_moe``, which sglang's caller looks up when it runs): the callee is
wrapped in its own module, as ``@vllm_route`` does."""

from __future__ import annotations

import inspect
import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason

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
    """Declare that sglang's ``caller`` reaches AKLR ops through this function where it calls
    ``callees`` (each as the dotted name ``caller`` looks up at run time), on the sglang
    releases ``versions`` it was tested on; on any other, sglang runs its own code."""

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
    untested = []
    for target, versions, route in _ROUTES:
        if installed in versions:
            # sglang only logs a target it cannot resolve
            pkgutil.resolve_name(target)
            HookRegistry.register(target, _around(route), HookType.AROUND)
        else:
            untested.append(target)
    calls = []
    for caller, callees, versions, route in _CALL_ROUTES:
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
                setattr(owner, attr, _around_call(route, orig))
            calls.append(callee)
    # a warning, so that the line shows at any sglang log level
    _LOGGER.warning(
        "aklr hooks %d sglang functions and %d calls on sglang %s%s",
        len(_ROUTES) - sum(t in untested for t, _, _ in _ROUTES),
        len(calls),
        installed,
        f"; not {', '.join(untested)}, untested on it" if untested else "",
    )


def _around(route: Callable) -> Callable:
    def around(orig, *args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            result = route(*args, **kwargs)
            if not isinstance(result, FallbackReason):
                return result
        return orig(*args, **kwargs)

    return around


def _around_call(route: Callable, orig: Callable) -> Callable:
    """``orig``, reaching ``route`` first: as ``_around``, for a callee wrapped in place."""

    def around(*args, **kwargs):
        if not torch.compiler.is_compiling():
            result = route(*args, **kwargs)
            if not isinstance(result, FallbackReason):
                return result
        return orig(*args, **kwargs)

    around.__wrapped__ = orig
    around.aklr_route = route
    return around
