# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""An op that ranks call together: its contract's ``ranks`` and each rank's inputs."""

import pytest
import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase


def _with_ranks(contract_id: str, ranks: str, group_op: str = "mega_moe@1.0") -> OpContract:
    """``contract_id``'s contract with ``ranks``, and ``group_op``'s reference and baseline."""
    registry = ContractRegistry.default()
    group = registry.get(group_op)
    return OpContract.model_validate(
        registry.get(contract_id).model_dump()
        | {"ranks": ranks, "reference": group.reference, "baseline": group.baseline}
    )


def test_ranks_at_reads_the_attr() -> None:
    case = OpCase.of("fp8_paged_mqa_logits@1.0")
    contract = _with_ranks("fp8_paged_mqa_logits@1.0", "page_size")
    assert contract.ranks_at(case.point) == case.point.attrs["page_size"]
    assert case.contract.ranks_at(case.point) == 1


def test_ranks_rejects_what_is_not_an_attr() -> None:
    """``ranks`` names an attr of positive ints."""
    with pytest.raises(ValueError, match="ranks: "):
        _with_ranks("fp8_paged_mqa_logits@1.0", "nope")


def test_ranks_needs_a_group_reference_and_baseline() -> None:
    """A contract with ``ranks`` has a DistOpReference and a DistOpBaseline."""
    with pytest.raises(ValueError, match="ranks: "):
        _with_ranks("fp8_paged_mqa_logits@1.0", "page_size", group_op="fp8_paged_mqa_logits@1.0")


def test_each_rank_draws_its_own_inputs() -> None:
    """Rank 0 draws a one-rank call's inputs; another rank its own, the same each time."""
    request = OpCase.of("gemm@1.0").request
    one_rank, rank0, rank1 = (
        request.compose_inputs(None, "cpu"),
        request.compose_inputs(None, "cpu", 0),
        request.compose_inputs(None, "cpu", 1),
    )
    assert rank0.keys() == rank1.keys()
    tensors = [n for n, v in rank0.items() if torch.is_tensor(v)]
    assert all(torch.equal(one_rank[n], rank0[n]) for n in tensors)
    assert not any(torch.equal(rank0[n], rank1[n]) for n in tensors)
    again = request.compose_inputs(None, "cpu", 1)
    assert all(torch.equal(again[n], rank1[n]) for n in tensors)
