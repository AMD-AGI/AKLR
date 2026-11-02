# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``deepseek_v4.embed``: the whole-table lookup writes the rows (zeros outside the
vocab) sglang's vocab-parallel embedding and ``hc_broadcast`` write, and ``replicate``
gathers the shards into that table."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu

VOCAB, DIM = 129280, 5120


def _table(n, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(n, d, generator=g, device="cuda").to(torch.bfloat16)


@pytest.mark.parametrize("rows,hc,dim", [(1, 4, DIM), (6, 4, DIM), (48, 4, DIM), (5, 1, 1000)])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_lookup_is_the_rows_broadcast(rows, hc, dim, dtype):
    from aklr.plugins.sglang.models.deepseek_v4 import embed

    table = _table(VOCAB, dim, rows)
    ids = torch.randint(0, VOCAB, (rows,), device="cuda")
    ids[0] = VOCAB - 1
    if rows > 2:
        ids[1], ids[2] = -1, VOCAB
    got = embed._lookup(table, ids.to(dtype), hc)
    valid = (ids >= 0) & (ids < VOCAB)
    want = torch.where(valid[:, None], table[ids.clamp(0, VOCAB - 1)], 0)
    want = want[:, None].expand(rows, hc, dim)
    assert got.shape == (rows, hc, dim)
    assert torch.equal(got.view(torch.int16), want.contiguous().view(torch.int16))


def test_lookup_without_a_table_is_none():
    from aklr.plugins.sglang.models.deepseek_v4 import embed

    ids = torch.zeros(3, dtype=torch.int64, device="cuda")
    assert embed.lookup(SimpleNamespace(), ids, 4) is None


@pytest.fixture
def group():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=0, world_size=1)
    cpu = dist.new_group(backend="gloo")
    yield SimpleNamespace(world_size=1, cpu_group=cpu, device_group=dist.group.WORLD)
    dist.destroy_process_group()


def test_replicate_builds_the_table(monkeypatch, group):
    from aklr.plugins.sglang import stats
    from aklr.plugins.sglang.models.deepseek_v4 import embed

    n, pad = 1000, 24
    w = _table(n + pad, 256)
    module = SimpleNamespace(
        weight=w,
        org_vocab_size=n,
        shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=n),
    )
    monkeypatch.setattr(embed, "_refusal", lambda module: None)
    monkeypatch.setattr(embed, "_input_scattered", lambda: False)
    copies = {}
    embed.replicate(module, group, copies)
    table = copies["embed table",]
    assert torch.equal(table.view(torch.int16), w[:n].view(torch.int16))
    ids = torch.tensor([0, n - 1, n, -1, 17], device="cuda")
    got = embed.lookup(module, ids, 4)
    assert torch.equal(got[:, 2], torch.cat([w[[0, n - 1]], w.new_zeros(2, 256), w[[17]]]))
    assert "whole table" in stats.snapshot()["info"][embed.INFO]


def test_refuses_when_off(monkeypatch):
    from aklr.plugins.sglang.models.deepseek_v4 import embed

    monkeypatch.setenv(embed.ENV_EMBED, "0")
    assert "=0" in embed._refusal(SimpleNamespace())
    monkeypatch.setenv(embed.ENV_EMBED, "2")
    with pytest.raises(ValueError):
        embed._refusal(SimpleNamespace())
