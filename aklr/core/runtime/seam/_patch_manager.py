###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable

from aklr.core.runtime.seam._adapter import Adapter

__all__ = ["PatchManager"]


class PatchManager:
    """Swaps entry points for their adapters' stand-ins in every loaded module, and back."""

    def __init__(self) -> None:
        self._patched: dict[str, tuple[Callable, Callable]] = {}  # entry point -> (original, stand-in)

    def patch(self, adapter: Adapter) -> bool:
        """Patch ``adapter.entry``; False if this environment lacks it or its backend fails to import."""
        if adapter.entry in self._patched:
            return True
        # shortcut: module-level functions only; resolve Class.method when a target needs it
        try:
            original = adapter.original()
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"{adapter.entry} is not served: its backend failed to import, {type(e).__name__}: {e}",
                stacklevel=3,
            )
            return False
        if original is None:
            return False
        stand_in = adapter.wrap(original)
        self._rebind(original, stand_in)
        self._patched[adapter.entry] = (original, stand_in)
        return True

    def unpatch_all(self) -> None:
        for original, stand_in in self._patched.values():
            self._rebind(stand_in, original)
        self._patched.clear()

    def _rebind(self, old: Callable, new: Callable) -> None:
        # every module's namespace: callers that ran `from x import f` hold their own f
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if type(namespace) is dict:
                namespace.update({n: new for n, v in list(namespace.items()) if v is old})
