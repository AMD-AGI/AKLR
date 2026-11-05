# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""An op that ranks call together: its contract's ``ranks`` and each rank's inputs."""

import pytest
import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase


def _with_ranks(contract_id: str, ranks: str) -> OpContract:
    contract = ContractRegistry.default().get(contract_id)
    return OpContract.model_validate(contract.model_dump() | {"ranks": ranks})


def test_ranks_at_reads_the_attr() -> None:
    case = OpCase.of("fp8_paged_mqa_logits@1.0")
    contract = _with_ranks("fp8_paged_mqa_logits@1.0", "page_size")
    assert contract.ranks_at(case.point) == case.point.attrs["page_size"]
    assert case.contract.ranks_at(case.point) == 1


@pytest.mark.parametrize(
    ("contract_id", "ranks"),
    [("fp8_paged_mqa_logits@1.0", "nope"), ("fused_moe@1.0", "act_quant")],
    ids=["not_an_attr", "baseline_relative"],
)
def test_ranks_rejects(contract_id: str, ranks: str) -> None:
    """``ranks`` names an attr of positive ints, and no bound needs the baseline in the
    reference task, which runs on one device."""
    with pytest.raises(ValueError, match="ranks: "):
        _with_ranks(contract_id, ranks)


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


def test_no_group_py_is_no_group_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """An op without ``group.py`` sets nothing up; one whose own import fails raises."""
    from aklr.core.contract import op

    contract = OpCase.of("gemm@1.0").contract
    assert contract.group_init() is None

    def unimportable(op_package: str, submodule: str) -> None:
        raise ModuleNotFoundError("No module named 'mori'", name="mori")

    monkeypatch.setattr(op, "op_module", unimportable)
    with pytest.raises(ModuleNotFoundError, match="mori"):
        contract.group_init()
