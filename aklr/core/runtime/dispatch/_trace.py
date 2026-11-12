# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``AKLR_TRACE``: a log line per hooked call, to see which ops a run reaches.

Two layers trace, each in its own format and logger: ``plugin``, every vLLM or sglang
route, and ``op``, every op api dispatch (the aklr/ops apis aiter or a route calls, and
the plugins' own ops). Each says whether AKLR served the call or why it fell back. A
process logs each distinct call once. A CUDA graph replay runs no Python, so it logs
nothing: decode shows its capture calls."""

from __future__ import annotations

import functools
import logging
import os
import threading
from collections.abc import Hashable
from typing import TYPE_CHECKING

from aklr.core.contract.fastkey import FastKey, FastKeyCodec
from aklr.core.contract.spec import axes_text
from aklr.core.platform.env import trace_mode
from aklr.core.runtime.builder import Runnable

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract
    from aklr.core.runtime.dispatch._dispatcher import FallbackReason

__all__ = ["Trace", "trace"]

_PLUGIN_LOGGER = logging.getLogger("aklr.trace.plugin")
_OP_LOGGER = logging.getLogger("aklr.trace.op")


class Trace:
    """The layers ``AKLR_TRACE`` turns on, and the calls already logged."""

    __slots__ = ("_lock", "_seen", "is_op", "is_plugin")

    def __init__(self, mode: str) -> None:
        self.is_plugin = mode in ("plugin", "all")
        self.is_op = mode in ("op", "all")
        self._seen: set[Hashable] = set()
        self._lock = threading.Lock()

    def _is_new(self, identity: Hashable) -> bool:
        with self._lock:
            if identity in self._seen:
                return False
            self._seen.add(identity)
            return True

    def plugin(self, framework: str, target: str, route: str, fallback: FallbackReason | None) -> None:
        """A route's call of ``target``: served if ``fallback`` is ``None``, else falling back
        to the framework's own code for it."""
        outcome, reason = ("served", "-") if fallback is None else ("fallback", fallback.value)
        if not self._is_new((framework, target, outcome, reason)):
            return
        # a warning, so that the line shows at any vLLM or sglang log level
        _PLUGIN_LOGGER.warning(
            "[aklr plugin] pid=%d framework=%s target=%s route=%s result=%s reason=%s",
            os.getpid(),
            framework,
            target,
            route,
            outcome,
            reason,
        )

    def op(self, contract: OpContract, key: FastKey | None, served: Runnable | FallbackReason) -> None:
        """A dispatch of ``contract`` at ``key`` (``None``: unkeyed), served by the runnable
        ``served``, else falling back for the ``FallbackReason`` ``served``."""
        if isinstance(served, Runnable):
            meta = served.metadata
            kernel = f"{meta.build_type}:{meta.schema_name}:{meta.content_hash[:12]}"
            outcome, reason = "served", "-"
        else:
            kernel, outcome, reason = "-", "fallback", served.value
        if not self._is_new((contract.contract_id, key, kernel, reason)):
            return
        scenario = point_text = "-"
        if key is not None:
            point = FastKeyCodec.from_contract(contract).to_point(key)
            scenario, point_text = contract.name_at(point), axes_text(point.axes)
        _OP_LOGGER.warning(
            "[aklr op] pid=%d contract=%s scenario=%s point=%s result=%s kernel=%s reason=%s",
            os.getpid(),
            contract.contract_id,
            scenario,
            point_text,
            outcome,
            kernel,
            reason,
        )


@functools.cache
def trace() -> Trace | None:
    """The process's trace, ``None`` while ``AKLR_TRACE`` is off; read once."""
    mode = trace_mode()
    return None if mode == "0" else Trace(mode)
