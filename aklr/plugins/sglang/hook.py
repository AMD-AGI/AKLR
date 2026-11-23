# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Install the SGLang function wrappers declared on its plugin."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version
from threading import RLock

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.runtime.dispatch._trace import trace

__all__ = ["install", "sglang_version"]

_LOGGER = logging.getLogger(__name__)

_lock = RLock()
_registry = None
_installed: set[str] = set()


def sglang_version() -> str:
    """The installed sglang's release, without a local label (0.5.22+rocm is 0.5.22)."""
    return Version(version("sglang")).public


def install() -> None:
    """Wrap the targets of the routes tested on the installed sglang."""
    with _lock:
        _install()


def _install() -> None:
    global _registry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from aklr.plugins.sglang import plugin

    routes = plugin.build()._wraps
    if _registry is not HookRegistry:
        _installed.clear()
        _registry = HookRegistry

    installed = sglang_version()
    untested, absent, pending = [], [], []
    for declaration in routes:
        target, versions, route = declaration.target, declaration.versions, declaration.callback
        if installed not in versions:
            untested.append(target)
            continue
        if target in _installed:
            continue
        try:
            # sglang only logs a target it cannot resolve
            original = pkgutil.resolve_name(target)
        except ModuleNotFoundError as e:
            # a target that needs a package this environment does not have (aiter off ROCm) is never called;
            # one missing from a package it has is a renamed target, which ends the start
            package = (e.name or "").partition(".")[0]
            if not package or importlib.util.find_spec(package) is not None:
                raise
            absent.append(f"{target} ({package} is not installed)")
            continue
        _validate_signature(target, original, route, declaration.around)
        pending.append((target, _around(route, target, declaration.around, declaration.eager)))
    for target, wrapper in pending:
        HookRegistry.register(target, wrapper, HookType.AROUND)
        _installed.add(target)
    # a warning, so that the line shows at any sglang log level
    _LOGGER.warning(
        "aklr hooks %d sglang functions on sglang %s%s%s",
        len(_installed),
        installed,
        f"; not {', '.join(untested)}, untested on it" if untested else "",
        f"; not {', '.join(absent)}" if absent else "",
    )


def _validate_signature(target: str, original: Callable, route: Callable, around: bool) -> None:
    """Check positional calls, keyword calls and calls that omit optional parameters."""
    params = inspect.signature(original).parameters.values()
    callback = inspect.signature(route)
    token = object()
    for keywords, required_only in ((False, False), (True, False), (True, True)):
        args = [original] if around else []
        kwargs = {}
        for p in params:
            if required_only and p.default is not p.empty:
                continue
            if p.kind == p.VAR_POSITIONAL:
                if not keywords:
                    args.extend((token, token))
            elif p.kind == p.VAR_KEYWORD:
                kwargs["__aklr_extra_keyword__"] = token
            elif p.kind == p.KEYWORD_ONLY or (keywords and p.kind == p.POSITIONAL_OR_KEYWORD):
                kwargs[p.name] = token
            else:
                args.append(token)
        try:
            callback.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{target}: {route.__qualname__} has incompatible parameters: {exc}") from exc


def _around(route: Callable, target: str, wraps_original: bool, eager: bool) -> Callable:
    route_name = f"{route.__module__}.{route.__qualname__}"

    def serve(orig, args, kwargs):
        if wraps_original:
            return route(orig, *args, **kwargs)
        result = route(*args, **kwargs)
        fallback = result if isinstance(result, FallbackReason) else None
        if (tracer := trace()) is not None and tracer.is_plugin:
            tracer.plugin("sglang", target, route_name, fallback)
        return orig(*args, **kwargs) if fallback is not None else result

    serve_eagerly = torch.compiler.disable(serve) if eager else None

    def around(orig, *args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            return serve(orig, args, kwargs)
        if serve_eagerly is not None:
            return serve_eagerly(orig, args, kwargs)
        return orig(*args, **kwargs)

    return around
