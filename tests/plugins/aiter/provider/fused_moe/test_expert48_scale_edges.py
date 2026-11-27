from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import ExpertMaskDistribution
from tests.op_case import OpCase


@pytest.fixture
def expert48_schema():
    path = Path(
        "aklr/plugins/aiter/provider/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts48_topk6_inter_dim3072_global_experts384/dbd3861a735b189f.json"
    )
    return ContractRegistry.default().parse_schema(path.read_bytes(), str(path))


def prepared(case, hidden, w1_scale, w2_scale, w2_code, route_weight):
    inputs = case.inputs()
    inputs["hidden_states"].fill_(hidden)
    inputs["w1"].view(torch.uint8).fill_(0x22)
    inputs["w2"].view(torch.uint8).fill_(w2_code)
    inputs["w1_scale"].view(torch.uint8).fill_(w1_scale)
    inputs["w2_scale"].view(torch.uint8).fill_(w2_scale)
    inputs["topk_ids"].copy_(
        torch.tensor([0, 100, 101, 102, 103, 104], dtype=torch.int32).expand(case.point.axes["n_tokens"], -1)
    )
    inputs["topk_weights"].fill_(route_weight)
    inputs["expert_mask"].zero_()
    inputs["expert_mask"][:48] = 1
    assert ExpertMaskDistribution(local_experts=case.point.axes["n_experts"]).contains(
        inputs["expert_mask"], case.point.dtypes["expert_mask"]
    )
    return inputs


@pytest.mark.gfx950
def test_minimum_e8m0_scale_has_nonzero_contribution(builder_registry, expert48_schema):
    case = OpCase(expert48_schema.point({"n_tokens": 16}))
    inputs = prepared(case, 0.625, 127 - 13, 0, 0x22, 2.0**120)
    expected = case.reference_outputs(inputs)["out"]
    assert torch.isfinite(expected).all() and (expected > 0).all()
    call = case.call(inputs, "cuda:0")
    assert call["w1"].is_shuffled and call["w2"].is_shuffled
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    builder_registry.build(expert48_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"].cpu(), expected, rtol=0.02, atol=0)


@pytest.mark.gfx950
def test_large_scaled_weights_with_zero_input_stay_zero(builder_registry, expert48_schema):
    case = OpCase(expert48_schema.point({"n_tokens": 16}))
    inputs = prepared(case, 0.0, 127, 254, 0x77, 1.0)
    call = case.call(inputs, "cuda:0")
    assert call["w1"].is_shuffled and call["w2"].is_shuffled
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    builder_registry.build(expert48_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"], torch.zeros_like(call["out"]), atol=0, rtol=0)


@pytest.mark.gfx950
def test_mixed_weight_scale_groups_match_finite_reference(builder_registry, expert48_schema):
    case = OpCase(expert48_schema.point({"n_tokens": 16}))
    inputs = prepared(case, 0.625, 114, 127, 0x22, 1.0)
    # Alternating signs and distinct scales in each 32-column group expose scale-group mixing.
    weights = inputs["w2"].view(torch.uint8)
    weights[..., :16] = 0x22
    weights[..., 16:32] = 0xAA
    scales = inputs["w2_scale"].view(torch.uint8)
    scales[..., 0::3] = 126
    scales[..., 1::3] = 128
    scales[..., 2::3] = 130
    expected = case.reference_outputs(inputs)["out"]
    assert torch.isfinite(expected).all() and (expected != 0).any()
    call = case.call(inputs, "cuda:0")
    assert call["w1"].is_shuffled and call["w2"].is_shuffled
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    builder_registry.build(expert48_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"].cpu(), expected, rtol=0.02, atol=0)
