# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""radix_routes' offset insert walk leaves sglang's unified radix tree as sglang's walk does:
the same insert results, nodes, device values and free slots after chunked-prefill chains,
splits, repeats and partial pages, on each of sglang's cache test configs."""

import dataclasses
import random

import pytest

pytest.importorskip("sglang.srt.mem_cache.unified_cache.unified_tree_core")
kits = pytest.importorskip("sglang.test.kits.unified_radix_cache.fixtures")

from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.kits.unified_radix_cache.config import CONFIGS
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures

from aklr.plugins.sglang.models.deepseek_v4 import radix_routes


def _configs():
    """sglang's configs with room for the requests (10 at most: the mamba configs' slots),
    and one bigram (EAGLE) key."""
    out = [
        dataclasses.replace(
            c,
            kv_size=max(c.kv_size, 128 * c.page_size),
            max_context_len=max(c.max_context_len, 16 * c.page_size),
        )
        for c in CONFIGS
    ]
    return out + [dataclasses.replace(out[1], is_eagle=True)]


def _requests(page: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    base = [rng.randrange(1, 50) for _ in range(12 * page)]
    out = [base[: k * 2 * page] for k in range(1, 6)]  # one node per prefill chunk
    out.append(base)
    out.append(base[: 5 * page] + [rng.randrange(50, 99) for _ in range(3 * page)])
    out.append(base[: 3 * page + page // 2] + [rng.randrange(50, 99) for _ in range(page)])
    out.append(base + [rng.randrange(50, 99) for _ in range(2 * page + 1)])
    out.append([rng.randrange(1, 50) for _ in range(4 * page)])
    return out


def _tree(cache, components) -> list:
    rows, todo = [], [(cache.root_node_handle(), 0)]
    while todo:
        node, depth = todo.pop()
        children = kits.node_children(cache, node)
        for child in sorted(children, key=lambda c: list(kits.node_token_ids(cache, c))):
            values = []
            for component in components:
                v = kits.node_device_value(cache, child, component)
                values.append(None if v is None else v.tolist())
            rows.append((depth, list(kits.node_token_ids(cache, child)), values))
            todo.append((child, depth + 1))
    return rows


def _run(cfg, seed: int):
    fx = UnifiedRadixCacheFixtures()
    fx.cfg = cfg
    cache, allocator, req_to_token_pool = kits.build_fixture(cfg, tree_core_backend="python")
    results = []
    for tokens in _requests(cfg.page_size, seed):
        results.append(fx._insert(cache, allocator, req_to_token_pool, tokens).prefix_len)
    return results, _tree(cache, cfg.components), allocator.available_size()


@pytest.mark.gpu  # sglang's cache fixtures allocate their pools on the GPU
@pytest.mark.parametrize("cfg", _configs(), ids=lambda c: c.label)
@pytest.mark.parametrize("seed", [0, 1])
def test_offset_walk_inserts_as_sglang(monkeypatch, cfg, seed):
    monkeypatch.setenv(radix_routes.ENV, "1")
    step = UnifiedTreeCore._insert_walk_step
    keys = []

    def walk_step(self, state):
        keys.append(len(state.key))
        return step(self, state)

    monkeypatch.setattr(UnifiedTreeCore, "_insert_walk_step", walk_step)
    expected = _run(cfg, seed)
    sglang_keys, keys[:] = list(keys), []
    orig = UnifiedTreeCore._advance_insert

    def routed(self):
        return radix_routes._advance_insert(orig, self)

    monkeypatch.setattr(UnifiedTreeCore, "_advance_insert", routed)
    assert _run(cfg, seed) == expected
    # the same steps, each given at most its node's head
    assert len(keys) == len(sglang_keys)
    assert all(a <= b for a, b in zip(keys, sglang_keys))
    assert sum(keys) < sum(sglang_keys)
