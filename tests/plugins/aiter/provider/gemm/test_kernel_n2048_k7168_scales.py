from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase


@pytest.fixture
def kernel_schema():
    path = Path(
        "aklr/plugins/aiter/provider/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n2048_k7168/282433dab1b9542b.json"
    )
    return ContractRegistry.default().parse_schema(path.read_bytes(), str(path))


@pytest.mark.gfx950
def test_zero_operands_with_large_valid_scales_stay_zero(builder_registry, kernel_schema):
    case = OpCase(kernel_schema.point({"m": 256}))
    inputs = case.inputs()
    inputs["a"].view(torch.uint8).zero_()
    inputs["b"].view(torch.uint8).zero_()
    inputs["a_scale"].fill_(2.0**64)
    inputs["b_scale"].fill_(2.0**64)
    call = case.call(inputs, "cuda:0")
    call["out"] = torch.full(
        (case.point.axes["m"], case.point.axes["n"]),
        float("nan"),
        dtype=getattr(torch, case.point.dtypes["out"]),
        device="cuda:0",
    )
    builder_registry.build(kernel_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"], torch.zeros_like(call["out"]), atol=0, rtol=0)


@pytest.mark.gfx950
def test_small_fp8_values_with_large_scales_have_finite_products(builder_registry, kernel_schema):
    case = OpCase(kernel_schema.point({"m": 256}))
    inputs = case.inputs()
    for key in ("a", "b"):
        code = torch.tensor([2.0**-9]).to(inputs[key].dtype).view(torch.uint8).item()
        inputs[key].view(torch.uint8).fill_(code)
    inputs["a_scale"].fill_(2.0**65)
    inputs["b_scale"].fill_(2.0**65)
    call = case.call(inputs, "cuda:0")
    call["out"] = torch.full(
        (case.point.axes["m"], case.point.axes["n"]),
        float("nan"),
        dtype=getattr(torch, case.point.dtypes["out"]),
        device="cuda:0",
    )
    builder_registry.build(kernel_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"], torch.full_like(call["out"], 7 * 2.0**122), atol=0, rtol=0)


def test_split_k_call_initializes_its_tag_words(builder_registry, kernel_schema, monkeypatch):
    import flydsl.compiler as compiler
    import flydsl.expr as expr

    case = OpCase(kernel_schema.point({"m": 256}))
    inputs = case.inputs()
    call = case.call(inputs)
    call["out"] = torch.empty(256, 2048, dtype=torch.bfloat16)
    runnable = builder_registry.build(kernel_schema)
    empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        value = empty(*args, **kwargs)
        return value.fill_(1)

    recorded = []

    def compiled(kernel, *args):
        recorded.append(args)
        return lambda *later: recorded.append(later)

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: object())
    monkeypatch.setattr(expr, "Stream", lambda stream: stream)
    class ExternalJit:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args):
            return compiler.compile(self, *args)

    monkeypatch.setattr(compiler, "jit", ExternalJit)
    monkeypatch.setattr(compiler, "compile", compiled)
    runnable(**call)
    tags = [args[2] for args in recorded]
    assert tags and all(t.numel() > 0 for t in tags)
    assert all(torch.count_nonzero(t).item() == 0 for t in tags)


@pytest.mark.gfx950
def test_split_k_state_is_reset_on_every_graph_replay(builder_registry, kernel_schema):
    case = OpCase(kernel_schema.point({"m": 256}))
    inputs = case.inputs()
    inputs["a"].view(torch.uint8).zero_()
    one = torch.tensor([1.0]).to(inputs["b"].dtype).view(torch.uint8).item()
    inputs["b"].view(torch.uint8).fill_(one)
    inputs["a_scale"].fill_(1.0)
    inputs["b_scale"].fill_(1.0)
    call = case.call(inputs, "cuda:0")
    call["out"] = torch.empty(256, 2048, dtype=torch.bfloat16, device="cuda:0")
    runnable = builder_registry.build(kernel_schema)
    runnable(**call)
    torch.cuda.synchronize(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runnable(**call)
    for value in (0.0, 1.0, 0.0, 2.0, -1.0):
        code = torch.tensor([value]).to(inputs["a"].dtype).view(torch.uint8).item()
        call["a"].view(torch.uint8).fill_(code)
        call["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(0)
        torch.testing.assert_close(call["out"], torch.full_like(call["out"], value * 7168), atol=0, rtol=0)

def test_calls_use_current_compiler_options(builder_registry, kernel_schema, monkeypatch):
    import os

    import flydsl.compiler as compiler
    import flydsl.expr as expr

    requested_options = []

    def compile_kernel(kernel, *args):
        requested_options.append(os.environ["FLYDSL_COMPILE_OPT_LEVEL"])

        def launch(*values):
            values[1].fill_(1)

        launch(*args)
        return launch

    class ExternalJit:
        def __init__(self, function):
            self.function = function
            self.specializations = {}

        def __call__(self, *args):
            key = os.environ["FLYDSL_COMPILE_OPT_LEVEL"]
            if key in self.specializations:
                self.specializations[key](*args)
            else:
                self.specializations[key] = compiler.compile(self, *args)

    monkeypatch.setattr(compiler, "jit", ExternalJit)
    monkeypatch.setattr(compiler, "compile", compile_kernel)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: object())
    monkeypatch.setattr(expr, "Stream", lambda value: value)
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "1")
    case = OpCase(kernel_schema.point({"m": 256}))
    call = case.call(case.inputs())
    call["out"] = torch.empty(256, 2048, dtype=torch.bfloat16)
    runnable = builder_registry.build(kernel_schema)
    empty = torch.empty

    def host_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    runnable(**call)
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "3")
    call["out"].fill_(float("nan"))
    runnable(**call)
    torch.testing.assert_close(call["out"], torch.ones_like(call["out"]), atol=0, rtol=0)
    assert requested_options == ["1", "3"]
