# Created by OpenAI
"""Sequence sampling keeps the actual host maximum and device partition coherent."""

import pytest
import torch

from aklr.core.contract.semantics import BoundedCuSeqlensDistribution


@pytest.mark.parametrize(
    "total,parts,maximum",
    [
        (4, 2, 2),
        (9, 3, 6),
        (512, 2, 256),
        (2049, 2, 2048),
        (8193, 5, 2048),
        (15361, 8, 2048),
        (15000, 3, 5000),
        (16384, 2, 8192),
        (16384, 2, 10000),
    ],
)
def test_partitions_have_the_declared_actual_maximum(total, parts, maximum):
    distribution = BoundedCuSeqlensDistribution(total=total, max_seqlen=maximum)
    for seed in range(5):
        sampled = distribution.sample((parts + 1,), "int32", torch.Generator().manual_seed(seed))
        lengths = sampled.diff().tolist()
        assert sampled[0] == 0 and sampled[-1] == total
        assert len(lengths) == parts and min(lengths) > 0
        assert max(lengths) == maximum
        assert distribution.contains(sampled, "int32")


@pytest.mark.parametrize("values", [[0, 2, 4, 6], [1, 3, 6, 9], [0, 0, 6, 9], [0, 4, 7, 9]])
def test_rejects_metadata_outside_the_stated_partition(values):
    distribution = BoundedCuSeqlensDistribution(total=9, max_seqlen=6)
    assert not distribution.contains(torch.tensor(values, dtype=torch.int32), "int32")


def test_rejects_an_impossible_maximum():
    distribution = BoundedCuSeqlensDistribution(total=4, max_seqlen=4)
    with pytest.raises(ValueError):
        distribution.sample((3,), "int32", torch.Generator())
