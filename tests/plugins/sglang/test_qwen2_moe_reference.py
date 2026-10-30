# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The qwen2_moe references: the routing they pick and the experts they sum."""

import torch

from aklr.core.contract import semantics
from aklr.plugins.sglang.models.qwen2_moe import reference
from tests.op_case import OpCase


def test_logits_round_once_to_bf16():
    # float32 first would land on the bf16 midpoint 1 + 2**-8 and round down to 1
    x = torch.tensor([1 + 2**-8 + 2**-40, -(1 + 2**-8 + 2**-40), 3.0, 0.0], dtype=torch.float64)
    rounded = reference._rounded_to_bf16(x)
    assert rounded.tolist() == [1 + 2**-7, -(1 + 2**-7), 3.0, 0.0]


def test_routing_takes_the_largest_logits_and_ties_go_to_the_lower_expert():
    logits = torch.tensor([[1.0, 3.0, 2.0, 3.0, -1.0, 2.0]], dtype=torch.float64)
    weights, ids = reference._routed(logits, 3)
    assert ids.tolist() == [[1, 3, 2]]
    # the softmax over every expert, renormalized over the chosen ones
    full = logits.softmax(-1)[0, [1, 3, 2]]
    torch.testing.assert_close(weights[0], full / full.sum())


def test_the_router_picks_by_bf16_logits():
    case = OpCase.of("qwen2_moe_router@1.0")
    args = case.inputs()
    weights, ids = case.reference_outputs(args).values()
    # two logits a hair apart in float64 tie in bf16: the lower expert wins
    hidden = torch.zeros_like(args["hidden_states"])
    hidden[:, 0] = 1
    router = torch.zeros_like(args["router_weight"])
    router[:, 0] = torch.linspace(-1, 1, router.shape[0])
    router[5, 0], router[9, 0] = 4.0, 4.0
    router[9, 1] = 2**-20  # hidden's column 1 is 0: no effect
    router[12, 0] = 4.0 + 2**-12  # rounds to bf16 4.0
    tied = {**args, "hidden_states": hidden, "router_weight": router}
    _, tied_ids = case.reference_outputs(tied).values()
    assert tied_ids[0, :3].tolist() == [5, 9, 12]
    assert ids.dtype == weights.dtype == torch.float64
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, dtype=torch.float64))


def test_the_block_is_the_experts_on_the_routers_routing():
    """qwen2_moe_fp8_router_experts' reference is qwen2_moe_fp8_experts' on
    qwen2_moe_router's routing."""
    case = OpCase.of("qwen2_moe_fp8_router_experts@1.0")
    args = case.inputs()
    (out,) = case.reference_outputs(args).values()
    router = OpCase.of("qwen2_moe_router@1.0")
    weights, ids = router.reference_outputs({n: args[n] for n in ("hidden_states", "router_weight")}).values()
    experts = OpCase.of("qwen2_moe_fp8_experts@1.0")
    routed = {**args, "topk_weights": weights.float(), "topk_ids": ids.int()}
    (expected,) = experts.reference_outputs({n: routed[n] for n in experts.inputs()}).values()
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-9)


def test_the_shared_expert_is_gated_by_the_sigmoid_of_its_gate():
    case = OpCase.of("qwen2_moe_fp8_experts@1.0")
    args = case.inputs()
    (out,) = case.reference_outputs(args).values()
    # routed experts at weight 0 leave the shared one alone; doubling its gate's logit
    # changes the output by the sigmoid's ratio
    shared_only = {**args, "topk_weights": torch.zeros_like(args["topk_weights"])}
    (shared,) = case.reference_outputs(shared_only).values()
    assert not torch.equal(shared, out) and shared.abs().sum() > 0
    x = args["hidden_states"].double()
    gate = (x @ args["shared_gate_weight"].double()[0]).sigmoid()
    doubled = {**shared_only, "shared_gate_weight": args["shared_gate_weight"] * 2}
    (twice,) = case.reference_outputs(doubled).values()
    ratio = (x @ args["shared_gate_weight"].double()[0] * 2).sigmoid() / gate
    torch.testing.assert_close(twice, shared * ratio[:, None])


def test_the_experts_quantize_each_linears_input_per_128():
    x = torch.randn(2, 256, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    quantized = reference._fp8_input(x).unflatten(-1, (-1, 128))
    groups = reference._rounded_to_bf16(x).unflatten(-1, (-1, 128))
    scale = groups.abs().amax(-1, keepdim=True).float() / 448
    # each group's values are fp8 codes of its scale, its amax the largest code
    codes = quantized / scale.double()
    assert torch.equal(codes, codes.float().to(torch.float8_e4m3fn).double())
    assert codes.abs().amax(-1).eq(448).all()
    # within half a code of each value: 2**-4 of a normal code, 2**-10 scales below
    torch.testing.assert_close(quantized, groups, rtol=2**-4, atol=1e-5)
    assert reference._fp8_input(torch.zeros(1, 128, dtype=torch.float64)).eq(0).all()


def test_the_references_leave_their_inputs_alone():
    for contract_id in (
        "qwen2_moe_router@1.0",
        "qwen2_moe_fp8_experts@1.0",
        "qwen2_moe_fp8_router_experts@1.0",
    ):
        case = OpCase.of(contract_id)
        args = case.inputs()
        before = {n: v.clone() for n, v in args.items() if torch.is_tensor(v)}
        reference_fn = getattr(reference, case.contract.reference_name)
        reference_fn(case.point, args, semantics.to_float64)
        for name, value in before.items():
            assert torch.equal(args[name].view(torch.uint8), value.view(torch.uint8)), name
