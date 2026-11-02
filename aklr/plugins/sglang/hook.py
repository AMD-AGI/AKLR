# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@sglang_route``: how a sglang function reaches AKLR ops, declared beside them."""

from __future__ import annotations

import logging
import pkgutil
from collections.abc import Callable
from importlib.metadata import version

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason

__all__ = ["install", "sglang_route", "sglang_version"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, tuple[str, ...], Callable]] = []


def sglang_route(target: str, *, versions: tuple[str, ...]) -> Callable[[Callable], Callable]:
    """Declare that sglang's ``target`` reaches AKLR ops through this function, on the sglang
    releases ``versions`` it was tested on; on any other, sglang runs its own code."""

    def declare(route: Callable) -> Callable:
        _ROUTES.append((target, versions, route))
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
    # a warning, so that the line shows at any sglang log level
    _LOGGER.warning(
        "aklr hooks %d sglang functions on sglang %s%s",
        len(_ROUTES) - len(untested),
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
