# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import struct

import pytest
import torch

from aklr.core.jit import hsaco
from tests.op_case import hsaco_code_object, hsaco_gemm_code_object

# reads blockDim and gridDim, which HIP passes as hidden arguments after out
_DIMS_DEVICE_SOURCE = """\
#include <hip/hip_runtime.h>

extern "C" __global__ void dims(int *out) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = blockDim.x * 1000 + gridDim.x;
}
"""


def _gemm(kernel: hsaco.Kernel, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    kernarg = struct.pack("<QQQii", a.data_ptr(), b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])
    kernel.launch((a.shape[0],), (b.shape[0],), kernarg)


@pytest.mark.gpu
class TestKernel:
    """``Kernel.launch`` runs a code object's function on the current stream."""

    def test_launches(self) -> None:
        kernel = hsaco.Kernel(hsaco_gemm_code_object(), "gemm")
        a, b = torch.randn(8, 32, device="cuda").bfloat16(), torch.randn(16, 32, device="cuda").bfloat16()
        out = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
        _gemm(kernel, a, b, out)
        torch.testing.assert_close(out.float(), a.float() @ b.float().mT, rtol=1e-2, atol=1e-2)

    def test_replays_in_a_cuda_graph(self) -> None:
        """A launch captured on a side stream replays with the arguments it was captured with."""
        kernel = hsaco.Kernel(hsaco_gemm_code_object(), "gemm")
        a, b = torch.randn(8, 32, device="cuda").bfloat16(), torch.randn(16, 32, device="cuda").bfloat16()
        out = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
        _gemm(kernel, a, b, out)  # loaded outside the capture, as a host warms up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _gemm(kernel, a, b, out)
        a.copy_(torch.randn_like(a))
        out.zero_()
        graph.replay()
        torch.testing.assert_close(out.float(), a.float() @ b.float().mT, rtol=1e-2, atol=1e-2)

    def test_shares_a_module_per_code_object(self) -> None:
        """Kernels of one code object load it once per device."""
        code_object = hsaco_gemm_code_object()
        kernels = hsaco.Kernel(code_object, "gemm"), hsaco.Kernel(code_object, "gemm")
        a, b = torch.randn(8, 32, device="cuda").bfloat16(), torch.randn(16, 32, device="cuda").bfloat16()
        for kernel in kernels:
            out = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
            _gemm(kernel, a, b, out)
            torch.testing.assert_close(out.float(), a.float() @ b.float().mT, rtol=1e-2, atol=1e-2)
        digest = kernels[0]._digest
        assert [key for key in hsaco._MODULES if key[1] == digest] == [(torch.cuda.current_device(), digest)]

    def test_launches_on_the_current_stream(self) -> None:
        """A launch under another stream runs on it."""
        kernel = hsaco.Kernel(hsaco_gemm_code_object(), "gemm")
        a, b = torch.randn(8, 32, device="cuda").bfloat16(), torch.randn(16, 32, device="cuda").bfloat16()
        out = torch.zeros(8, 16, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            _gemm(kernel, a, b, out)
        # had it run on the default stream, it would be done before the sleep
        assert out.eq(0).all()
        stream.synchronize()
        torch.testing.assert_close(out.float(), a.float() @ b.float().mT, rtol=1e-2, atol=1e-2)

    def test_fills_in_the_hidden_arguments(self) -> None:
        """A kernarg of the explicit arguments only: HIP fills in the hidden ones."""
        kernel = hsaco.Kernel(hsaco_code_object(_DIMS_DEVICE_SOURCE), "dims")
        out = torch.zeros(4 * 64, device="cuda", dtype=torch.int32)
        kernel.launch((4,), (64,), struct.pack("<Q", out.data_ptr()))
        assert out.eq(64 * 1000 + 4).all()

    def test_unknown_symbol(self) -> None:
        kernel = hsaco.Kernel(hsaco_gemm_code_object(), "gemv")
        with pytest.raises(RuntimeError, match="finding gemv in its code object"):
            kernel.launch((1,), (1,), b"")


def test_is_available_on_rocm_only() -> None:
    assert hsaco.is_available() == (torch.version.hip is not None)
