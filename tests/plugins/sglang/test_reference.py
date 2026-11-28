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


def test_latent_moe_front_fp32_rounds_the_latent_input_alone() -> None:
    """kimi_k3_latent_moe at front_fp32 (sglang 0.5.24's ROCm front): the router's logits and the shared experts' gate
    and up stay fp32. On one-hot rows the front is front_weight's bf16 values, the same either way; on any other
    rows the shared experts read other values than off the bf16 front."""
    case = OpCase.of("kimi_k3_latent_moe@1.0")
    fp32 = OpCase(case.point.model_copy(update={"attrs": {**case.point.attrs, "front_fp32": True}}))
    args = case.inputs()
    one_hot = torch.zeros_like(args["hidden_states"])
    one_hot[torch.arange(one_hot.shape[0]), torch.arange(one_hot.shape[0])] = 1
    rows = {**args, "hidden_states": one_hot}
    for name, value in case.reference_outputs(rows).items():
        assert torch.equal(value, fp32.reference_outputs(rows)[name]), name
    assert not torch.equal(case.reference_outputs(args)["shared"], fp32.reference_outputs(args)["shared"])
