# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@sglang_route``: how a sglang method reaches an op, declared beside the op's api."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import version

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, SpecifierSet, Callable]] = []


def sglang_route(target: str, versions: str) -> Callable[[Callable], Callable]:
    """Declare that sglang's ``target`` reaches an op through this function, on the sglang
    ``versions``: a range with an upper bound, since a later sglang may change the
    target."""
    # Parsed here so a bad range fails its own package's import, not every route.
    supported = SpecifierSet(versions)
    if not any(s.operator in ("<", "<=", "==", "===", "~=") for s in supported):
        raise ValueError(f"{target}: sglang versions {versions!r} have no upper bound")

    def declare(route: Callable) -> Callable:
        _ROUTES.append((target, supported, route))
        return route

    return declare


def install() -> None:
    """Wrap each declared target with its one route for the running sglang; a route
    returning a ``FallbackReason`` runs the original, one that raises is not caught."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    running = version("sglang")
    # A dev or rc build counts as its release: sglang images ship 0.5.19.dev<date>.
    release = Version(running).base_version
    declared: dict[str, list[tuple[SpecifierSet, Callable]]] = {}
    for target, supported, route in _ROUTES:
        declared.setdefault(target, []).append((supported, route))
    for target, routes in declared.items():
        matching = [r for s, r in routes if s.contains(release)]
        if len(matching) == 1:
            wrapper = _around(matching[0])
        else:
            ranges = "; ".join(str(supported) for supported, _ in routes)
            wrapper = _unrouted(
                f"[aklr] {target}: {len(matching)} routes for sglang {running}, "
                f"not 1 (declared: {ranges}); falling back to sglang"
            )
        HookRegistry.register(target, wrapper, HookType.AROUND)


def _unrouted(message: str) -> Callable:
    """The original, after one warning on first use: a layer this sglang runs without
    AKLR is never silent."""
    warned = False

    def around(orig, *args, **kwargs):
        nonlocal warned
        if not warned:
            warned = True
            _LOGGER.warning(message)
        return orig(*args, **kwargs)

    return around


def _around(route: Callable) -> Callable:
    def around(orig, *args, **kwargs):
        result = route(*args, **kwargs)
        return orig(*args, **kwargs) if isinstance(result, FallbackReason) else result

    return around
