# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What every selector of the provider shares: its registration, the candidate it offers and
the one-time start of AKLR's dispatch."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason

__all__ = ["NAME", "SELECTORS", "declining", "is_declining", "offer", "ready", "selector", "served"]

NAME = "aklr"
_QUALIFICATION = "https://github.com/AMD-AGI/AKLR"
_LOGGER = logging.getLogger("aklr.plugins.aiter")

Selector = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]
# aiter's op name -> (the contract version served, the selector)
SELECTORS: dict[str, tuple[int, Selector]] = {}
_is_ready: bool | None = None
_declining = threading.local()


def selector(op: str, version: int = 1) -> Callable[[Selector], Selector]:
    """Register the selector of aiter's ``op`` at contract ``version``. A selector takes the
    call as aiter hands it (every positional parameter, defaults applied, and the keyword-only
    ones), unpacks it by name and returns :func:`offer`'s candidate, or ``None`` to decline."""

    def register(fn: Selector) -> Selector:
        if op in SELECTORS:
            raise ValueError(f"aiter op {op!r} already has a selector")
        SELECTORS[op] = (version, fn)
        return fn

    return register


def ready() -> bool:
    """AKLR's dispatch, started on the first call aiter asks about; a process that cannot start
    it declines every call."""
    global _is_ready
    if _is_ready is None:
        try:
            DispatcherRegistry.default().init_once()
            _is_ready = True
        except Exception as e:  # noqa: BLE001 (AKLR's failure must not fail aiter's call)
            _LOGGER.warning("aklr: serves no aiter op in this process: %r", e)
            _is_ready = False
    return _is_ready


def offer(serving: RunnableMetadata | FallbackReason, run: Callable[..., Any]) -> Any:
    """The candidate running ``run`` when AKLR serves the call (``serving``, an api's
    ``serving`` answer, is the kernel), else ``None``."""
    if isinstance(serving, FallbackReason):
        return None
    from aiter import plugin

    return plugin.KernelCandidate(
        provider=NAME,
        kernel_id=serving.kernel_ref,
        artifact_digest=f"aklr:{serving.content_hash}",
        qualification=_QUALIFICATION,
        run=run,
    )


def served(result: Any) -> Any:
    """An api's result in ``run``; select has asked the same table, so a fallback is a bug,
    and aiter does not fall back once ``run`` started."""
    if isinstance(result, FallbackReason):
        raise RuntimeError(f"aklr: an accepted aiter call fell back ({result.value})")
    return result


@contextlib.contextmanager
def declining() -> Iterator[None]:
    """AKLR declines the aiter calls made in this block on this thread: a framework route that asked
    AKLR and fell back runs aiter's own kernel, without its call being offered to AKLR again."""
    previous = getattr(_declining, "active", False)
    _declining.active = True
    try:
        yield
    finally:
        _declining.active = previous


def is_declining() -> bool:
    """Whether this thread is in :func:`declining`."""
    return getattr(_declining, "active", False)
