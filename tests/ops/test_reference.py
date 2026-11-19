# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.validation.stages._bench._evaluation._comparison import (
    is_reference_comparable,
)
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
    """The reference's outputs are as declared, admitted by the contract's compare."""

    def test_outputs_are_declared(self, op_case: OpCase) -> None:
        """On the CPU, ``reference_outputs`` holds every result finite, in float32, or in its
        declared dtype for an integral output."""
        reference = op_case.reference_outputs(op_case.inputs())
        assert list(reference) == list(op_case.contract.mutable_tensors)
        for name, tensor in reference.items():
            dtype = op_case.point.dtypes[name]
            is_integral = semantics.DTYPE_TRAITS[dtype].is_integral
            assert tensor.device.type == "cpu", name
            assert tensor.dtype == (semantics.torch_dtypes(dtype)[0] if is_integral else torch.float32), name
            assert tensor.isfinite().all(), name
            assert tuple(tensor.shape) == op_case.contract.dtype_storage_shape_of(name, op_case.point), name

    def test_rounded_reference_passes_compare(self, op_case: OpCase) -> None:
        """The reference rounded to each output's dtype passes ``op_case.compare``."""
        args = op_case.inputs()
        assert is_reference_comparable(
            op_case.point.dtypes,
            op_case.reference_outputs(args),
            lambda rounded: op_case.compare(rounded, args, baseline_outputs=rounded),
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("op_case", OpCase.params(_backend_points()))
    def test_backend_matches_reference(self, op_case: OpCase) -> None:
        """``check_backend_matches_reference`` for aklr/ops, on the aiter aiter-test installs; a
        plugin's contracts in the job that times them (tests/plugins/<framework>/test_reference.py)."""
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
    from aklr.ops.gemm.reference import gemm_a8w8_reference

    point = SimpleNamespace(
        dtypes={"a": "int8", "b": "int8", "a_scale": "float32", "b_scale": "float32"},
        attrs={"trans_b": True},
    )
    a = torch.zeros(1, 128, dtype=torch.int8)
    b = torch.zeros(1, 128, dtype=torch.int8)
    a[0, :2] = torch.tensor([3, 5], dtype=torch.int8)
    b[0, :2] = torch.tensor([5, -3], dtype=torch.int8)
    args = {"a": a, "b": b, "a_scale": torch.tensor([[0.7]]), "b_scale": torch.tensor([[0.9]])}
    assert gemm_a8w8_reference(point, args).tolist() == [[0.0]]


def test_biased_grouped_topk_ranks_tied_experts_lower_first() -> None:
    """Equal biased scores: the lower expert ranks first, whatever torch.topk's algorithm does."""
    from aklr.ops.topk.reference import biased_grouped_topk_reference

    point = SimpleNamespace(
        dtypes={"router_logits": "float32", "correction_bias": "float32", "topk_ids": "int32"},
        axes={"topk": 3},
        attrs={"renormalize": False},
    )
    logits = torch.zeros(1, 1024)
    logits[0, [583, 571, 368, 31]] = 1.0
    args = {
        "router_logits": logits,
        "correction_bias": torch.zeros(1024),
        "routed_scaling_factor": 1.0,
    }
    _, ids = biased_grouped_topk_reference(point, args)
    assert ids.tolist() == [[31, 368, 571]]
