# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@sglang_route``: how a sglang method reaches an op, declared beside the op's api."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import version
from typing import NamedTuple

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import stats

__all__ = ["RouteFallback", "install", "installed_routes", "sglang_route"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, SpecifierSet, Callable, bool]] = []
_INSTALLED: dict[str, str] = {}  # target -> "routed" or why not


class RouteFallback(NamedTuple):
    """A route's answer that sglang's own method runs this call, and why: ``reason`` is a
    short stable key (counted, logged once), ``detail`` the call's specifics."""

    reason: str
    detail: str = ""


def sglang_route(target: str, versions: str, *, around: bool = False) -> Callable[[Callable], Callable]:
    """Declare that sglang's ``target`` reaches an op through this function, on the sglang
    ``versions``: a range with an upper bound, since a later sglang may change the
    target. With ``around``, the route is a step around sglang's method rather than an
    op: it is called as ``route(orig, *args, **kwargs)``, returns the method's result,
    and nothing is counted."""
    # Parsed here so a bad range fails its own package's import, not every route.
    supported = SpecifierSet(versions)
    if not any(s.operator in ("<", "<=", "==", "===", "~=") for s in supported):
        raise ValueError(f"{target}: sglang versions {versions!r} have no upper bound")

    def declare(route: Callable) -> Callable:
        _ROUTES.append((target, supported, route, around))
        return route

    return declare


def install() -> None:
    """Wrap each declared target with its one route for the running sglang; a route
    returning a ``FallbackReason`` or a ``RouteFallback`` runs the original, one that
    raises is not caught."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    running = version("sglang")
    # A dev or rc build counts as its release: sglang images ship 0.5.19.dev<date>.
    release = Version(running).base_version
    declared: dict[str, list[tuple[SpecifierSet, Callable, bool]]] = {}
    for target, supported, route, around in _ROUTES:
        declared.setdefault(target, []).append((supported, route, around))
    for target, routes in declared.items():
        matching = [(r, a) for s, r, a in routes if s.contains(release)]
        if len(matching) == 1:
            route, around = matching[0]
            wrapper = route if around else _around(target, route)
            _INSTALLED[target] = "routed"
        else:
            ranges = "; ".join(str(supported) for supported, *_ in routes)
            why = f"{len(matching)} routes for sglang {running}, not 1 (declared: {ranges})"
            wrapper = _unrouted(target, why)
            _INSTALLED[target] = why
        HookRegistry.register(target, wrapper, HookType.AROUND)
    unrouted = {t: why for t, why in sorted(_INSTALLED.items()) if why != "routed"}
    _LOGGER.info("%d sglang methods routed", len(_INSTALLED) - len(unrouted))
    if unrouted:
        _LOGGER.warning(
            "sglang runs its own code for %d methods:\n  - %s",
            len(unrouted),
            "\n  - ".join(f"{t}: {why}" for t, why in unrouted.items()),
        )


def installed_routes() -> dict[str, str]:
    """target -> "routed", or why sglang's own method runs it."""
    return dict(_INSTALLED)


def _unrouted(target: str, why: str) -> Callable:
    """The original, counted (``install`` warns once about every unrouted target): a layer
    this sglang runs without AKLR is never silent."""

    def around(orig, *args, **kwargs):
        stats.record(target, "fallback", "unrouted", why)
        return orig(*args, **kwargs)

    return around


def _around(target: str, route: Callable) -> Callable:
    def around(orig, *args, **kwargs):
        result = route(*args, **kwargs)
        if isinstance(result, RouteFallback):
            stats.record(target, "fallback", result.reason, result.detail)
            return orig(*args, **kwargs)
        if isinstance(result, FallbackReason):
            stats.record(target, "fallback", result.value)
            return orig(*args, **kwargs)
        stats.record(target, "served")
        return result

    return around
