# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The unified radix cache's insert walk in time linear in the key: each step gets only the
head of the key its node can match, and the rest is sliced once (sglang slices per node).
``AKLR_DSV41_RADIX_WALK=0`` keeps sglang's walk."""

from __future__ import annotations

import os

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV = "AKLR_DSV41_RADIX_WALK"
CORE = "sglang.srt.mem_cache.unified_cache.unified_tree_core"

SGLANG_READS = {
    "srt/mem_cache/unified_cache/unified_tree_core.py": (
        "UnifiedTreeCore",
        "_advance_insert",
        "_insert_walk_step(state)",
        "_is_deferrable_action",
        "_ongoing_insert_walk_state",
        "_InsertPhase",
        "WALK",
        "InsertStepResult",
        "phase",
        "pending_actions",
        "key",
        "node",
        "children",
        "page_size",
    ),
    "srt/mem_cache/radix_cache.py": (
        "RadixKey",
        "child_key_at",
        "__getitem__",
        "__len__",
    ),
}


def _enabled() -> bool:
    value = os.environ.get(ENV) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: expected '1' or '0'")
    return value == "1"


@sglang_route(f"{CORE}.UnifiedTreeCore._advance_insert", versions=VERSIONS, around=True)
def _advance_insert(orig, self):
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        InsertStepResult,
        _InsertPhase,
    )

    state = self._ongoing_insert_walk_state
    if not serving() or not _enabled() or state.phase is not _InsertPhase.WALK:
        return orig(self)
    stats.set_info("radix_insert_walk", "offset walk")
    key, page, off = state.key, self.page_size, 0
    while state.phase is _InsertPhase.WALK:
        rest = len(key) - off
        if rest < max(page, 1):
            break
        child_key = key.child_key_at(off, page)
        if child_key not in state.node.children:
            break
        n = min(len(state.node.children[child_key].key), rest)
        state.key = key[off : off + n]
        flushed = len(state.pending_actions)
        self._insert_walk_step(state)
        off += n - len(state.key)
        new = state.pending_actions[flushed:]
        if new and not all(map(self._is_deferrable_action, new)):
            state.key = key[off:]
            actions, state.pending_actions = state.pending_actions, []
            return InsertStepResult(actions=actions)
    state.key = key[off:] if off else key
    return orig(self)
