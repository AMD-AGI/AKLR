# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The SGLang indexer can identify which query rows share a speculative request."""

from contextlib import contextmanager
from contextvars import ContextVar

_Hint = tuple[int, int]
# Empty tuple marks an active indexer scope with no outstanding verify call.
_scoped: ContextVar[_Hint | tuple[()] | None] = ContextVar("aklr_verify_scope", default=None)
_legacy: ContextVar[_Hint | None] = ContextVar("aklr_verify_note", default=None)


@contextmanager
def target_verify(note: _Hint | None):
    token = _scoped.set(note or ())
    try:
        yield
    finally:
        _scoped.reset(token)


def note_target_verify(rows: int, draft_tokens: int) -> None:
    """Compatibility entry for integrations which announce one logits call."""
    _legacy.set((rows, draft_tokens) if rows else None)


def current() -> _Hint | None:
    scoped = _scoped.get()
    return _legacy.get() if scoped is None else scoped or None


def consume() -> None:
    if _scoped.get() is None:
        _legacy.set(None)
    else:
        _scoped.set(())
