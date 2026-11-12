# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.runtime.builder import BuilderRegistry, BuildError
from aklr.core.schema import Build, Kernel, Schema, SourceFile, Tensor, Var
from tests.op_case import OpCase

_GEMM = "def run(a, b, bias, out):\n    out.copy_(a @ b.mT)\n"
_GEMM_RETURNING = "def run(a, b, bias):\n    return a @ b.mT\n"
_GEMM_EITHER = "def run(a, b, bias, out=None):\n    return a @ b.mT if out is None else out.copy_(a @ b.mT)\n"
_HIP_GEMM = """\
#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <optional>
#include <torch/library.h>

namespace {

__global__ void gemm(const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *out,
                     int k) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a[blockIdx.x * k + i]) * __bfloat162float(b[threadIdx.x * k + i]);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = __float2bfloat16(acc);
}

void run(const at::Tensor &a, const at::Tensor &b, const std::optional<at::Tensor> &bias,
         at::Tensor &out) {
    gemm<<<a.size(0), b.size(0), 0, at::hip::getCurrentHIPStream()>>>(
        static_cast<const __hip_bfloat16 *>(a.data_ptr()),
        static_cast<const __hip_bfloat16 *>(b.data_ptr()),
        static_cast<__hip_bfloat16 *>(out.data_ptr()), static_cast<int>(a.size(1)));
}

} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, Tensor? bias, Tensor(a!) out) -> ()", &run);
}
"""
_HIP_SUM_AND_DIFFERENCE = """\
#include <ATen/ATen.h>
#include <torch/library.h>
#include <tuple>

namespace {

std::tuple<at::Tensor, at::Tensor> run(const at::Tensor &a, const at::Tensor &b) {
    return {a + b, a - b};
}

} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b) -> (Tensor, Tensor)", &run);
}
"""


def _kernel(source: str, *, language: str = "triton", dps: bool = True) -> Kernel:
    entry = "kernel.hip" if language == "hip" else "kernel.py"
    return Kernel(
        name=f"{language}_{'dps' if dps else 'returning'}",
        author="tests",
        build=Build(
            language=language,
            target_hardware=["gfx950"],
            entry_point=f"{entry}::run",
            destination_passing_style=dps,
        ),
        sources=[SourceFile(path=entry, content=source)],
    )


def _sum_and_difference(dps: bool) -> Schema:
    return Schema(
        op_type="sum_and_difference",
        axes={"m": Var()},
        inputs={
            "a": Tensor(["m"], dtype="float32"),
            "b": Tensor(["m"], dtype="float32"),
        },
        outputs={
            "x": Tensor(["m"], dtype="float32"),
            "y": Tensor(["m"], dtype="float32"),
        },
        kernel=_kernel(_HIP_SUM_AND_DIFFERENCE, language="hip", dps=dps),
    )


class TestBuilderRegistry:
    def test_runs_a_destination_passing_kernel(self, builder_registry: BuilderRegistry) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        out = torch.empty(4, 32, dtype=torch.bfloat16)
        builder_registry.build(case.schema(_kernel(_GEMM)))(call["a"], call["b"], None, out)
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_runs_a_value_returning_kernel(self, builder_registry: BuilderRegistry) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        runnable = builder_registry.build(case.schema(_kernel(_GEMM_RETURNING, dps=False)))
        assert torch.equal(runnable(call["a"], call["b"], None), call["a"] @ call["b"].mT)

    def test_keeps_the_calling_convention_of_each_kernel(self, builder_registry: BuilderRegistry) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        built = {
            dps: builder_registry.build(case.schema(_kernel(_GEMM_EITHER, dps=dps))) for dps in (True, False)
        }
        assert {dps: runnable.metadata.destination_passing_style for dps, runnable in built.items()} == {
            True: True,
            False: False,
        }

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            pytest.param("raise ImportError('no kernel')\n", "kernel", id="import"),
            pytest.param("def run(a, b):\n    pass\n", "positional", id="arity"),
        ],
    )
    def test_build_error(self, builder_registry: BuilderRegistry, source: str, match: str) -> None:
        with pytest.raises(BuildError, match=match):
            builder_registry.build(OpCase.from_id("gemm_bf16_nt").schema(_kernel(source)))

    def test_bad_aklr_value_is_not_a_build_error(
        self, builder_registry: BuilderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKLR_MIN_SPEEDUP", "abc")
        with pytest.raises(ValueError, match="AKLR_MIN_SPEEDUP"):
            builder_registry.build(OpCase.from_id("gemm_bf16_nt").schema(_kernel(_GEMM)))


@pytest.mark.gpu
class TestTorchBuilder:
    def test_runs_the_entry_op(self, builder_registry: BuilderRegistry) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call(device="cuda")
        out = torch.empty(4, 32, dtype=torch.bfloat16, device="cuda")
        builder_registry.build(case.schema(_kernel(_HIP_GEMM, language="hip")))(
            call["a"], call["b"], None, out
        )
        torch.testing.assert_close(
            out.float(), call["a"].float() @ call["b"].float().mT, rtol=1e-2, atol=1e-2
        )

    def test_runs_a_value_returning_op_of_two_outputs(self, builder_registry: BuilderRegistry) -> None:
        a, b = torch.arange(4.0, device="cuda"), torch.ones(4, device="cuda")
        x, y = builder_registry.build(_sum_and_difference(dps=False))(a, b)
        assert x.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert y.tolist() == [-1.0, 0.0, 1.0, 2.0]

    def test_op_not_taking_the_outputs_is_a_build_error(self, builder_registry: BuilderRegistry) -> None:
        with pytest.raises(BuildError, match="positional"):
            builder_registry.build(_sum_and_difference(dps=True))

    def test_kernel_that_does_not_compile_is_a_build_error(self, builder_registry: BuilderRegistry) -> None:
        source = _HIP_GEMM.replace("float acc = 0.0f;", "float acc = ;")
        with pytest.raises(BuildError, match="build"):
            builder_registry.build(OpCase.from_id("gemm_bf16_nt").schema(_kernel(source, language="hip")))
