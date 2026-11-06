# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A causal sequence's top-k slots: the rows sglang's DSA indexer hands its sparse attention."""

import pytest
import torch

from aklr.core.contract.semantics import CausalTopkDistribution


@pytest.mark.parametrize("rows,k,prefix", [(96, 64, 0), (40, 64, 0), (50, 64, 30), (300, 64, 100)])
def test_rows_name_their_visible_positions(rows, k, prefix):
    distribution = CausalTopkDistribution(hi=1023, prefix=prefix)
    sampled = distribution.sample((rows, k), "int32", torch.Generator().manual_seed(rows + prefix))
    assert sampled.dtype == torch.int32 and distribution.contains(sampled, "int32")
    counts = (sampled >= 0).sum(-1)
    seen = torch.arange(rows) + prefix + 1
    assert counts.equal(seen.clamp(max=k))
    # -1 only after a row's slots
    assert ((sampled >= 0) == (torch.arange(k) < counts[:, None])).all()
    # every slot one of the sequence's run of prefix + rows slots
    slots = sampled[sampled >= 0]
    assert slots.max() - slots.min() < prefix + rows
    base = int(sampled[0, 0]) if prefix == 0 else None
    for r in range(rows):
        row = sampled[r, : counts[r]]
        if seen[r] <= k and base is not None:
            assert row.equal(torch.arange(base, base + r + 1, dtype=torch.int32))


def test_full_rows_change_slowly():
    """Positions keep one key for the whole sequence, so a row adds the new token's position or
    keeps its neighbour's slots."""
    sampled = CausalTopkDistribution(hi=4095).sample((400, 64), "int32", torch.Generator().manual_seed(0))
    for r in range(65, 400):
        before, after = set(sampled[r - 1].tolist()), set(sampled[r].tolist())
        assert len(before - after) <= 1


def test_a_seed_draws_the_same_rows():
    """Bench compares two runs of one input set bit for bit; a row's order is part of it."""
    distribution = CausalTopkDistribution(hi=1 << 20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        first, second = (
            distribution.sample((3000, 2048), "int32", torch.Generator(device).manual_seed(12288))
            for _ in range(2)
        )
    assert first.equal(second)


@pytest.mark.parametrize(
    "values,inside",
    [
        ([[3, 1, -1, -1]], True),
        ([[-1, -1, -1, -1]], True),
        ([[3, 3, -1, -1]], False),
        ([[3, 8, -1, -1]], False),
        ([[3, -2, -1, -1]], False),
    ],
)
def test_contains_distinct_slots_and_any_number_of_minus_ones(values, inside):
    distribution = CausalTopkDistribution(hi=7)
    assert distribution.contains(torch.tensor(values, dtype=torch.int32), "int32") is inside


def test_rejects_a_sequence_longer_than_the_pool():
    with pytest.raises(ValueError):
        CausalTopkDistribution(hi=63, prefix=8).sample((60, 16), "int32", torch.Generator())
