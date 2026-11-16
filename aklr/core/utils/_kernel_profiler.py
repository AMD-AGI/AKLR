# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel benchmark sessions backed by AKLR's graph node attribution.

One session owns the collector until finalized. Its tree contains explicit Python
scopes and kernel leaves; CPU op tracing and a second GPU profiler are unnecessary.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

from aklr.core.utils.testing import kernel_tracer


class _Session:
    def __init__(self, name):
        self.name = name
        self.tracer = kernel_tracer()
        self.active = False
        self.complete = False
        self.local = threading.local()
        self.nodes = {0: {"frame": {"name": "ROOT"}, "children": []}}
        self.scopes = {}
        self.open_scopes = 0

    def deactivate(self):
        if not self.active:
            return
        if self.open_scopes:
            raise RuntimeError("cannot deactivate a kernel profiler session inside a scope")
        self.active = False
        # Drain work before stopping collection, including work on forked streams.
        import torch

        try:
            torch.cuda.synchronize()
        finally:
            records = self.tracer.stop()
        for scope_id, name, nanoseconds, exec_id, node_id, groups in records:
            self.nodes[scope_id]["children"].append(
                {
                    "frame": {"name": name},
                    "metrics": {"time (ns)": nanoseconds},
                    "attributes": {"graph exec": exec_id, "graph node": node_id, "groups": groups},
                }
            )
        self.complete = True


_session = None
_lifecycle_lock = threading.RLock()


def start(name, *, context="shadow", data="tree"):
    with _lifecycle_lock:
        global _session
        if context != "shadow" or data != "tree":
            raise ValueError("the kernel profiler supports context='shadow', data='tree' only")
        if _session is not None:
            raise RuntimeError("a kernel profiler session is already open")
        session = _Session(name)
        session.tracer.start()
        session.active = True
        _session = session
        return session


@contextmanager
def scope(name):
    session = _session
    if session is None or not session.active:
        raise RuntimeError("a kernel profiler scope requires an active session")
    stack = getattr(session.local, "stack", ())
    parent = stack[-1] if stack else 0
    key = parent, name
    if key not in session.scopes:
        scope_id = len(session.nodes)
        session.scopes[key] = scope_id
        node = {"frame": {"name": name}, "children": []}
        session.nodes[scope_id] = node
        session.nodes[parent]["children"].append(node)
    scope_id = session.scopes[key]
    session.tracer.push_scope(scope_id)
    session.local.stack = (*stack, scope_id)
    session.open_scopes += 1
    try:
        yield
    finally:
        try:
            session.tracer.pop_scope()
        finally:
            session.local.stack = stack
            session.open_scopes -= 1


def deactivate(session, *, flushing=True):
    with _lifecycle_lock:
        if session is not _session:
            raise RuntimeError("the kernel profiler session is not open")
        if not flushing:
            raise ValueError("kernel timing requires flushing the session")
        session.deactivate()


def get_data(session):
    if not session.complete:
        raise RuntimeError("kernel profiler data requires a successfully deactivated session")
    return [session.nodes[0]]


def finalize(session):
    with _lifecycle_lock:
        global _session
        if session is not _session:
            raise RuntimeError("the kernel profiler session is not open")
        if session.open_scopes:
            raise RuntimeError("cannot finalize a kernel profiler session inside a scope")
        try:
            session.deactivate()
            if session.complete:
                Path(f"{session.name}.hatchet").write_text(json.dumps(get_data(session)))
        finally:
            _session = None
