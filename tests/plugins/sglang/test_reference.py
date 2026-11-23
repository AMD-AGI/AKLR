# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

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
    assert case.compare({**reference, "topk_ids": swapped}, args)["topk_ids"].is_passed

    scores = case.contract.id_scores(case.point, [args], 0)["topk_ids"].float()
    assert scores[0, ids[0, 0]] - scores[0, ids[0, -1]] > 0.25  # more than two bf16 values below 16
    far = ids.clone()
    far[0, [0, -1]] = ids[0, [-1, 0]]
    assert not case.compare({**reference, "topk_ids": far}, args)["topk_ids"].is_passed


def test_sparse_mla_fp8_ds_prefill_reads_no_key_at_minus_one() -> None:
    """dsa_sparse_mla_fp8_ds_prefill: a token whose one key stands beside -1s attends that key alone, as one whose
    entries all name it does, its sum of exp(score) topk times smaller; a token of -1s only gets 0 and -inf."""
    case = OpCase.of("dsa_sparse_mla_fp8_ds_prefill@1.0")
    args = case.inputs()
    topk = args["indices"].shape[1]
    alone, repeated, empty = (args["indices"].clone() for _ in range(3))
    alone[0, 1:] = -1
    repeated[0, :] = alone[0, 0]
    empty[0, :] = -1
    outputs = [case.reference_outputs({**args, "indices": i}) for i in (alone, repeated, empty)]
    assert torch.allclose(outputs[0]["output"][0], outputs[1]["output"][0])
    assert torch.allclose(outputs[1]["lse"][0] - outputs[0]["lse"][0], torch.tensor(math.log(topk)))
    assert bool((outputs[2]["output"][0] == 0).all()) and bool((outputs[2]["lse"][0] == -math.inf).all())
