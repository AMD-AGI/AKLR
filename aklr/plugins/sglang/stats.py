# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What each sglang route did: served or fell back, and why. A CUDA graph replay never
reaches Python, so a route serving in captured graphs also keeps a device counter. With
``AKLR_EXPORT_DIR`` set, every process writes ``route_calls_<pid>.json`` there."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path

from aklr.core.platform.env import ENV_EXPORT_DIR

__all__ = ["device_counter", "dump", "record", "set_info", "snapshot"]

_LOGGER = logging.getLogger(__name__)
_DUMP_EVERY_S = 10.0

_lock = threading.Lock()
_counts: Counter[tuple[str, str, str]] = Counter()  # (target, outcome, reason)
_details: dict[tuple[str, str, str], str] = {}
_devices: dict[str, object] = {}  # name -> 1-element int64 device tensor
_info: dict[str, object] = {}
_last_dump = 0.0


def record(target: str, outcome: str, reason: str = "", detail: str = "") -> None:
    """Count one call of ``target``: ``outcome`` "served" or "fallback" (with its
    reason). The first fallback of each (target, reason) logs a warning."""
    key = (target, outcome, reason)
    with _lock:
        is_new = key not in _counts
        _counts[key] += 1
        if is_new and detail:
            _details[key] = detail
    if is_new and outcome == "fallback":
        _LOGGER.warning(
            "%s falls back to sglang: %s%s",
            target,
            reason,
            f" ({detail})" if detail else "",
        )
    if is_new or time.monotonic() - _last_dump > _DUMP_EVERY_S:
        dump()


def device_counter(name: str, device) -> object | None:
    """A 1-element int64 tensor on ``device``, counted as ``name`` in the export; a
    captured ``counter.add_(1)`` counts every replay of its graph. None when first asked
    for inside a capture: a tensor made there would be zeroed by every replay."""
    import torch

    with _lock:
        if name not in _devices:
            if _capturing():
                return None
            _devices[name] = torch.zeros(1, dtype=torch.int64, device=device)
        return _devices[name]


def existing_device_counter(name: str, device) -> object | None:
    """``device_counter``'s tensor where it is made (on ``device``), else None."""
    with _lock:
        t = _devices.get(name)
    return t if t is not None and t.device == device else None


def _capturing() -> bool:
    import torch

    # a process that never used the GPU has no graph, and is not made to open one here
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def set_info(key: str, value: object) -> None:
    """A static fact for the export, e.g. the layers a route covers."""
    with _lock:
        _info[key] = value


def snapshot() -> dict:
    """The counts so far; device counters read only outside a graph capture."""
    devices = {}
    if not _capturing():
        devices = {name: int(t.item()) for name, t in list(_devices.items())}
    with _lock:
        calls = [
            {
                "target": t,
                "outcome": o,
                "reason": r,
                "detail": _details.get((t, o, r), ""),
                "calls": n,
            }
            for (t, o, r), n in sorted(_counts.items())
        ]
        info = dict(_info)
    return {"pid": os.getpid(), "time": time.time(), "calls": calls,
            "device_counters": devices, "info": info}  # fmt: skip


def dump() -> None:
    """Write the snapshot to ``AKLR_EXPORT_DIR``, if set; never raises."""
    global _last_dump
    _last_dump = time.monotonic()
    directory = os.environ.get(ENV_EXPORT_DIR)
    if not directory:
        return
    try:
        data = snapshot()
        path = Path(directory) / f"route_calls_{os.getpid()}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str))
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001 -- statistics never stop serving
        _LOGGER.debug("cannot write route statistics: %s", e)


atexit.register(dump)
