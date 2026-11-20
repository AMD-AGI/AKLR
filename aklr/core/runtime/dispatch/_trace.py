# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.


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
        outcome, reason = ("served", "-") if fallback is None else ("fallback", fallback.value)
        if not self._is_new((framework, target, outcome, reason)):
            return
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
        if isinstance(served, Runnable):
            kernel, outcome, reason = served.metadata.kernel_ref, "served", "-"
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
    mode = trace_mode()
    return None if mode == "0" else Trace(mode)
