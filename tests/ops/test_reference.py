# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import torch

from aklr.core.contract import semantics
from tests.op_case import OpCase
from tests.ops import cases as ops_cases


def _backend_points():
    """Scope this one baseline/reference comparison; keep every variant of each affected contract."""
    points = ops_cases.SMALL_POINTS
    contracts = set(os.environ.get("AKLR_REFERENCE_CONTRACTS", "").split())
    if not contracts or not contracts.issubset({point.contract_id for point in points}):
        return points
    return [point for point in points if point.contract_id in contracts]


class TestReference:
    """The reference's outputs are as declared."""

    def test_outputs_are_declared(self, op_case: OpCase) -> None:
        """On the CPU, ``reference_outputs`` holds every result finite, in its declared dtype;
        what its check reads besides comes after them."""
        mutable_tensors = list(op_case.contract.mutable_tensors)
        reference = op_case.reference_outputs(op_case.inputs())
        assert list(reference)[: len(mutable_tensors)] == mutable_tensors
        for name in mutable_tensors:
            tensor = reference[name]
            assert tensor.device.type == "cpu", name
            assert tensor.dtype == semantics.torch_dtypes(op_case.point.dtypes[name])[0], name
            assert tensor.float().isfinite().all(), name
            assert tuple(tensor.shape) == op_case.contract.dtype_storage_shape_of(name, op_case.point), name

    @pytest.mark.gpu
    @pytest.mark.parametrize("op_case", OpCase.params(_backend_points()))
    def test_backend_matches_reference(self, op_case: OpCase) -> None:
        """``check_backend_matches_reference`` for aklr/ops, on the aiter aiter-test installs; a
        plugin's contracts in the job that times them (tests/plugins/<framework>/test_reference.py)."""
        if op_case.contract_id in {"mla_kc_rope_cache@1.0", "situv2_a4w4_moe@1.0"}:
            pytest.skip(
                "fails since #318 put it here: aiter's kernel does not compile, or is off the reference"
            )
        op_case.check_backend_matches_reference()


@pytest.mark.parametrize("contract_id", ["fp8_paged_mqa_logits@1.0", "fp8_paged_mqa_logits_verify@1.0"])
def test_mqa_logits_reference_is_the_same_in_chunks(
    monkeypatch: pytest.MonkeyPatch, contract_id: str
) -> None:
    """One row a chunk gives the paged MQA logits references bit for bit."""
    import aklr.ops.mla.reference as mla_reference

    case = OpCase.of(contract_id)
    args = case.inputs()
    whole = case.reference_outputs(args)["logits"]
    monkeypatch.setattr(mla_reference, "_CHUNK_ELEMENTS", 1)
    assert torch.equal(case.reference_outputs(args)["logits"], whole)


def test_int8_gemm_sums_its_products_exactly() -> None:
    """An int8 dot of 0 (3 * 5 + 5 * -3) scaled by 0.7 and 0.9 is 0: the int32 sum, then the
    scales, not a sum of scaled products."""
    from aklr.ops.gemm.reference import GemmA8w8Reference

    a = torch.zeros(1, 128, dtype=torch.int8)
    b = torch.zeros(1, 128, dtype=torch.int8)
    a[0, :2] = torch.tensor([3, 5], dtype=torch.int8)
    b[0, :2] = torch.tensor([5, -3], dtype=torch.int8)
    args = {
        "a": a,
        "b": b,
        "a_scale": torch.tensor([[0.7]]),
        "b_scale": torch.tensor([[0.9]]),
        "trans_b": True,
        "out_dtype": torch.bfloat16,
    }
    assert GemmA8w8Reference().reference(**args)["out"].tolist() == [[0.0]]


def test_gemm_reference_returns_out_in_out_dtype() -> None:
    """A bf16 gemm into a float32 out (Kimi-K3's router) is float32, not a's dtype."""
    from aklr.ops.gemm.reference import GemmReference

    a = torch.ones(2, 128, dtype=torch.bfloat16)
    b = torch.ones(3, 128, dtype=torch.bfloat16)
    out = GemmReference().reference(a=a, b=b, trans_b=True, out_dtype=torch.float32)["out"]
    assert out.dtype == torch.float32


def test_biased_grouped_topk_ranks_tied_experts_lower_first() -> None:
    """Equal biased scores: the lower expert ranks first, whatever torch.topk's algorithm does."""
    from aklr.ops.topk.reference import BiasedGroupedTopkReference

    logits = torch.zeros(1, 1024)
    logits[0, [583, 571, 368, 31]] = 1.0
    args = {
        "router_logits": logits,
        "correction_bias": torch.zeros(1024),
        "routed_scaling_factor": 1.0,
        "renormalize": False,
        "topk_ids": torch.empty(1, 3, dtype=torch.int32),
    }
    ids = BiasedGroupedTopkReference().reference(**args)["topk_ids"]
    assert ids.tolist() == [[31, 368, 571]]


@pytest.mark.parametrize("contract_id", ["topk_transform_paged@1.0", "dsa_topk_transform@1.0"])
def test_a_full_row_of_selected_slots_may_come_in_any_order(contract_id: str) -> None:
    """A row holding no -1 may list its slots in any order; a row padded with -1 keeps its order,
    and a full row with another slot is still wrong."""
    case = OpCase.of(contract_id)
    expected = torch.tensor([[3, 1, 2, 0], [5, 4, -1, -1]], dtype=torch.int32)

    def passes(out: list[list[int]]) -> bool:
        outputs = {"out_page_indices": torch.tensor(out, dtype=torch.int32)}
        return (
            case.contract.reference.check(outputs, {"out_page_indices": expected}, case.request.schema) == {}
        )

    assert passes([[0, 1, 2, 3], [5, 4, -1, -1]])
    assert not passes([[3, 1, 2, 0], [4, 5, -1, -1]])
    assert not passes([[3, 1, 2, 7], [5, 4, -1, -1]])
