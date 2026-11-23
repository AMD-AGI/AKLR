# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What every selector of the provider shares: its registration, the candidate it offers and
the one-time start of AKLR's dispatch."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason

__all__ = ["NAME", "declining", "is_declining", "offer", "ready", "served"]

NAME = "aklr"
_QUALIFICATION = "https://github.com/AMD-AGI/AKLR"
_LOGGER = logging.getLogger("aklr.plugins.aiter")

_is_ready: bool | None = None
_init_lock = threading.Lock()
_declining = threading.local()


def ready() -> bool:
    """AKLR's dispatch, started on the first call aiter asks about; a process that cannot start
    it declines every call."""
    global _is_ready
    if _is_ready is not None:
        return _is_ready
    import torch

    registry = DispatcherRegistry.default()
    if not registry.is_initialized and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    with _init_lock:
        if _is_ready is not None:
            return _is_ready
        try:
            registry.init_once()
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
