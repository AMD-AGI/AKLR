# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract import semantics
from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM
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


def _glm53_rank_args(args: dict, point, rank: int, tp: int) -> dict:
    """``rank``'s args in a group of ``tp``: its share of ``args``' heads and expert intermediates,
    the rest whole."""
    ax = point.axes
    heads, inter = ax["n_heads"] // tp, ax["inter_dim"] // tp
    q_rows, o_cols = heads * (ax["qk_nope_dim"] + ROPE_DIM), heads * ax["v_head_dim"]

    def share(tensor, size, dim, blocks=1):
        size //= blocks
        return tensor.narrow(dim, rank * size, size).contiguous()

    def gate_up(tensor, size, blocks=1):
        gate, up = tensor.chunk(2, dim=1)
        return torch.cat([share(gate, size, 1, blocks), share(up, size, 1, blocks)], dim=1)

    return {
        **args,
        "q_b_weight": share(args["q_b_weight"], q_rows, 0),
        "q_b_scale": share(args["q_b_scale"], q_rows, 0, 128),
        "w_kc": share(args["w_kc"], heads, 0),
        "w_vc": share(args["w_vc"], heads, 0),
        "o_weight": share(args["o_weight"], o_cols, 1),
        "o_scale": share(args["o_scale"], o_cols, 1, 128),
        "w1": gate_up(args["w1"], inter),
        "w1_scale": gate_up(args["w1_scale"], inter, 128),
        "w2": share(args["w2"], inter, 2),
        "w2_scale": share(args["w2_scale"], inter, 2, 128),
    }


def test_glm53_moe_layer_reference_over_two_ranks_is_one_ranks():
    """Heads and expert intermediates split over two ranks in whole 128-blocks, the rest on both: each
    rank's out is the one rank's, but for the sums' order, and so is its KV pool."""
    case = OpCase.of("glm53_moe_layer@1.0")
    args, contract = case.inputs(), case.contract
    whole = contract.reference_outputs(case.point, [args], 0, semantics.to_float32)
    ax = case.point.axes
    point = case.point.model_copy(
        update={
            "axes": {**ax, "n_heads": ax["n_heads"] // 2, "inter_dim": ax["inter_dim"] // 2},
            "attrs": {**case.point.attrs, "tp_size": 2},
        }
    )
    group = [_glm53_rank_args(args, case.point, rank, 2) for rank in range(2)]
    for rank in range(2):
        split = contract.reference_outputs(point, group, rank, semantics.to_float32)
        torch.testing.assert_close(split["out"], whole["out"], rtol=1e-4, atol=1e-4)
        assert torch.equal(split["kv_cache"], whole["kv_cache"])
