# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The hsaco gemm: a code object hipcc builds here and the kernel sources that launch it."""

from __future__ import annotations

import base64
import functools
import subprocess
import tempfile
from pathlib import Path

import torch

HSACO_GEMM_DEVICE = """\
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void gemm(const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *out,
                                int n, int k) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a[blockIdx.x * k + i]) * __bfloat162float(b[threadIdx.x * k + i]);
    }
    out[blockIdx.x * n + threadIdx.x] = __float2bfloat16(acc);
}
"""
HSACO_GEMM_KERNEL = '''\
"""out = a @ b.T in bf16 from a code object, a thread an output element."""

import struct

from aklr.core.jit import hsaco

from .codeobjects import GEMM

_GEMM = hsaco.Kernel(GEMM, "gemm")


def run(a, b, bias, out):
    kernarg = struct.pack("<QQQii", a.data_ptr(), b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])
    _GEMM.launch((a.shape[0],), (b.shape[0],), kernarg, device=a.device)
'''


@functools.cache
def hsaco_code_object(device_source: str = HSACO_GEMM_DEVICE) -> bytes:
    """device_source built by ROCm's hipcc into a code object for the GPU's arch."""
    from torch.utils import cpp_extension

    arch = torch.cuda.get_device_properties(0).gcnArchName.partition(":")[0]
    with tempfile.TemporaryDirectory() as tmp:
        source, code_object = Path(tmp, "kernel.hip"), Path(tmp, "kernel.hsaco")
        source.write_text(device_source)
        hipcc = Path(cpp_extension.ROCM_HOME, "bin", "hipcc")
        subprocess.run(  # noqa: S603 -- the toolchain's own compiler on a source of this file
            [hipcc, "--genco", f"--offload-arch={arch}", "-O3", source, "-o", code_object],
            check=True,
            capture_output=True,
        )
        return code_object.read_bytes()


def hsaco_gemm_sources() -> dict[str, str]:
    """HSACO_GEMM_KERNEL with its codeobjects.py, which holds the gemm's code object as GEMM."""
    encoded = base64.b64encode(hsaco_code_object()).decode()
    return {
        "kernel.py": HSACO_GEMM_KERNEL,
        "codeobjects.py": f'"""The gemm."""\n\nimport base64\n\nGEMM = base64.b64decode("{encoded}")\n',
    }
