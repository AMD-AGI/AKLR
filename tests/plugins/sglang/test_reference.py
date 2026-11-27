# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from tests.op_case import OpCase
from tests.plugins.sglang import cases


@pytest.mark.gpu
@pytest.mark.parametrize("op_case", OpCase.params(cases.BACKEND_POINTS))
def test_backend_matches_reference(op_case: OpCase) -> None:
    """``check_backend_matches_reference`` in sglang-test: sglang's call on the aiter its image ships."""
    op_case.check_backend_matches_reference()


def test_router_ids_of_equal_logits_may_swap() -> None:
    """qwen2_moe_router: experts 0 and 1 share a router row, so their bf16 logits are equal and
    either may come first; swapping a row's first and last chosen experts still fails."""
    case = OpCase.of("qwen2_moe_router@1.0")
    args = case.inputs()
    args["router_weight"][1] = args["router_weight"][0]
    reference = case.reference_outputs(args)
    ids = reference["topk_ids"]
    rows = ((ids == 0).any(-1) & (ids == 1).any(-1)).nonzero()[:, 0]
    assert len(rows) > 0
    swapped = ids.clone()
    for r in rows.tolist():
        swapped[r][ids[r] == 0], swapped[r][ids[r] == 1] = 1, 0
    assert "topk_ids" not in case.compare({**reference, "topk_ids": swapped}, args)

    scores = reference["topk_id_scores"].float()
    assert scores[0, ids[0, 0]] - scores[0, ids[0, -1]] > 0.25  # more than two bf16 values below 16
    far = ids.clone()
    far[0, [0, -1]] = ids[0, [-1, 0]]
    assert "topk_ids" in case.compare({**reference, "topk_ids": far}, args)


@pytest.mark.parametrize("contract_id", ["dsa_indexer_key_prepare@1.0", "dsa_indexer_prepare@1.0"])
def test_index_k_cache_key_at_twice_the_scale_stands_for_the_reference(contract_id: str) -> None:
    """A key stored at twice the reference's scale, its fp8 values half, passes; at four times the
    scale, or at twice it with the values left as they are, it still fails."""
    import torch

    from aklr.plugins.sglang.layers.dsa_indexer.reference import HEAD_DIM, PAGE_SIZE, _key_columns

    case = OpCase.of(contract_id)
    args = case.inputs()
    reference = case.reference_outputs(args)
    page, values, scales = (c[:1] for c in _key_columns(reference["written_slots"]))

    def stored_at(factor: float, value_factor: float) -> dict:
        pages = reference["index_k_cache"].clone().view(torch.uint8).reshape(-1, PAGE_SIZE * (HEAD_DIM + 4))
        scale = pages[page, scales].contiguous().view(torch.float32) * factor
        pages[page, scales] = scale.view(torch.uint8)
        codes = pages[page, values].view(torch.float8_e4m3fn).float() * value_factor
        pages[page, values] = codes.to(torch.float8_e4m3fn).view(torch.uint8)
        return {
            **reference,
            "index_k_cache": pages.view(torch.float8_e4m3fn).reshape(args["index_k_cache"].shape),
        }

    def passes(outputs: dict) -> bool:
        return "index_k_cache" not in case.compare(outputs, args)

    assert passes(stored_at(2.0, 0.5))
    assert not passes(stored_at(4.0, 0.25))
    assert not passes(stored_at(2.0, 1.0))
