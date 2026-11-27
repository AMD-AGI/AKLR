import pytest
import torch

from tests.op_case import OpCase


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize(
    "hidden_value,w1_exponent,w2_exponent",
    [(2.0**-40, 28, -9), (1.0, -32, 30)],
    ids=["stage1_tiny", "stage2_tiny"],
)
def test_small_moe_tiny_activations_round_to_zero(
    builder_registry, small_moe_schema, rows, hidden_value, w1_exponent, w2_exponent
):
    case = OpCase(small_moe_schema.point({"n_tokens": rows}))
    inputs = case.inputs()
    inputs["hidden_states"].fill_(hidden_value)
    inputs["w1"].view(torch.uint8).fill_(0x22)
    inputs["w2"].view(torch.uint8).fill_(0x22)
    inputs["w1_scale"].view(torch.uint8).fill_(127 + w1_exponent)
    inputs["w2_scale"].view(torch.uint8).fill_(127 + w2_exponent)
    inputs["topk_ids"].copy_(torch.arange(9, dtype=torch.int32).expand(rows, -1))
    inputs["topk_weights"].fill_(1.0)
    call = case.call(inputs, "cuda:0")
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    builder_registry.build(small_moe_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"], torch.zeros_like(call["out"]), atol=0, rtol=0)
