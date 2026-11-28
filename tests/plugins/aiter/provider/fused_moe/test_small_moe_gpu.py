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


@pytest.mark.gfx950
def test_small_moe_uses_the_tensor_device_nondefault_stream(builder_registry, small_moe_schema):
    assert torch.cuda.device_count() >= 2
    original_device = torch.cuda.current_device()
    saved_streams = {i: torch.cuda.current_stream(i) for i in (0, 1)}
    try:
        torch.empty(1, device="cuda:1")
        torch.cuda.synchronize(1)
        torch.cuda.set_device(0)
        case = OpCase(small_moe_schema.point({"n_tokens": 1}))
        inputs = case.inputs()
        inputs["hidden_states"].zero_()
        inputs["w1"].view(torch.uint8).zero_()
        inputs["w2"].view(torch.uint8).zero_()
        call = case.call(inputs, "cuda:0")
        call["out"] = torch.empty_like(call["hidden_states"])
        runnable = builder_registry.build(small_moe_schema)
        runnable(**call)
        matrix = torch.ones(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
        matrix @ matrix
        torch.cuda.synchronize(0)
        stream = torch.cuda.Stream(device=0)
        torch.cuda.set_stream(stream)
        with torch.cuda.stream(stream):
            for _ in range(16):
                matrix @ matrix
            call["out"].fill_(float("nan"))
        torch.cuda.set_device(1)
        runnable(**call)
        stream.synchronize()
        torch.cuda.synchronize(1)
        torch.testing.assert_close(call["out"], torch.zeros_like(call["out"]), atol=0, rtol=0)
    finally:
        for device, previous in saved_streams.items():
            torch.cuda.set_device(device)
            torch.cuda.set_stream(previous)
        torch.cuda.set_device(original_device)
