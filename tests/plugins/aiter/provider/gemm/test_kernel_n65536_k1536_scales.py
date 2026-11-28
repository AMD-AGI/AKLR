from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase


@pytest.fixture
def kernel_schema():
    path = Path(
        "aklr/plugins/aiter/provider/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/60b7d4855c2d5afe.json"
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
def test_valid_scale_ratios_do_not_overflow(builder_registry, kernel_schema):
    case = OpCase(kernel_schema.point({"m": 256}))
    inputs = case.inputs()
    inputs["a"].view(torch.uint8).zero_()
    inputs["b"].view(torch.uint8).zero_()
    for key in ("a", "b"):
        code = torch.tensor([1.0]).to(inputs[key].dtype).view(torch.uint8).item()
        inputs[key].view(torch.uint8)[:, 0].fill_(code)
    for key in ("a_scale", "b_scale"):
        inputs[key].fill_(1.0)
        inputs[key][:, 0].fill_(2.0**40)
        inputs[key][:, 1].fill_(2.0**-40)
    call = case.call(inputs, "cuda:0")
    call["out"] = torch.full(
        (case.point.axes["m"], case.point.axes["n"]),
        float("nan"),
        dtype=getattr(torch, case.point.dtypes["out"]),
        device="cuda:0",
    )
    builder_registry.build(kernel_schema)(**call)
    torch.cuda.synchronize(0)
    torch.testing.assert_close(call["out"], torch.full_like(call["out"], 2.0**80), atol=0, rtol=0)


def test_calls_use_current_compiler_options(builder_registry, kernel_schema, monkeypatch):
    import os

    import flydsl.compiler as compiler
    import flydsl.expr as expr

    requested_options = []

    def compile_kernel(kernel, *args):
        requested_options.append(os.environ["FLYDSL_COMPILE_OPT_LEVEL"])

        def launch(*values):
            values[2].fill_(1)

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
    call["out"] = torch.empty(256, 65536, dtype=torch.bfloat16)
    runnable = builder_registry.build(kernel_schema)
    runnable(**call)
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "3")
    call["out"].fill_(float("nan"))
    runnable(**call)
    torch.testing.assert_close(call["out"], torch.ones_like(call["out"]), atol=0, rtol=0)
    assert requested_options == ["1", "3"]
