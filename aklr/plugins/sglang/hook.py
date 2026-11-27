# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@sglang_route``: how a sglang function reaches AKLR ops, declared beside them."""

from __future__ import annotations

import importlib.util
import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.runtime.dispatch._trace import trace

__all__ = ["install", "sglang_route", "sglang_version", "sglang_version_matches"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, tuple[str, ...], Callable]] = []
# routes that run even where Dynamo compiles their caller, outside its graph
_EAGER: set[Callable] = set()


def sglang_route(
    target: str, *, versions: tuple[str, ...], eager: bool = False
) -> Callable[[Callable], Callable]:
    """Declare that sglang's ``target`` reaches AKLR ops through this function, on the sglang
    versions matching ``versions`` (exact releases or PEP 440 ranges). Outside that policy,
    sglang runs its own code. A range admits future versions without claiming they were tested. Where Dynamo
    compiles the caller, a route hands the call to sglang, unless ``eager``: one whose target's own
    code cannot run what another route set up then runs outside the compiled graph."""

    def declare(route: Callable) -> Callable:
        _ROUTES.append((target, versions, route))
        if eager:
            _EAGER.add(route)
        return route

    return declare


def sglang_version() -> str:
    """The installed sglang's release, without a local label (0.5.22+rocm is 0.5.22)."""
    return Version(version("sglang")).public


def sglang_version_matches(installed: str, versions: tuple[str, ...]) -> bool:
    """Match an exact release or any declared PEP 440 range, including daily builds."""
    return any(
        SpecifierSet(v if v.startswith(("<", ">", "=", "!", "~")) else f"=={v}").contains(
            installed, prereleases=True
        )
        for v in versions
    )


def install() -> None:
    """Wrap targets whose declared version policy admits the installed sglang."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    installed = sglang_version()
    untested, absent = [], []
    for target, versions, route in _ROUTES:
        if not sglang_version_matches(installed, versions):
            untested.append(target)
            continue
        try:
            # sglang only logs a target it cannot resolve
            pkgutil.resolve_name(target)
        except ModuleNotFoundError as e:
            # a target that needs a package this environment does not have (aiter off ROCm) is never called;
            # one missing from a package it has is a renamed target, which ends the start
            package = (e.name or "").partition(".")[0]
            if not package or importlib.util.find_spec(package) is not None:
                raise
            absent.append(f"{target} ({package} is not installed)")
            continue
        HookRegistry.register(target, _around(route, target), HookType.AROUND)
    # a warning, so that the line shows at any sglang log level
    _LOGGER.warning(
        "aklr hooks %d sglang functions on sglang %s%s%s",
        len(_ROUTES) - len(untested) - len(absent),
        installed,
        f"; not {', '.join(untested)}, outside their version policy" if untested else "",
        f"; not {', '.join(absent)}" if absent else "",
    )


def _around(route: Callable, target: str) -> Callable:
    route_name = f"{route.__module__}.{route.__qualname__}"

    def serve(orig, args, kwargs):
        result = route(*args, **kwargs)
        fallback = result if isinstance(result, FallbackReason) else None
        if (tracer := trace()) is not None and tracer.is_plugin:
            tracer.plugin("sglang", target, route_name, fallback)
        return orig(*args, **kwargs) if fallback is not None else result

    serve_eagerly = torch.compiler.disable(serve) if route in _EAGER else None

    def around(orig, *args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            return serve(orig, args, kwargs)
        if serve_eagerly is not None:
            return serve_eagerly(orig, args, kwargs)
        return orig(*args, **kwargs)

    return around
