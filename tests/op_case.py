# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import base64
import contextlib
import functools
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.contract.baseline import DistOpBaseline
from aklr.core.contract.fastkey import CallArgs
from aklr.core.contract.locate import op_dir
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import SOURCE_SUFFIXES, Language, PointSpec
from aklr.core.validation.stages._bench._evaluation._request import TrialRequest
from tests.ops import cases as ops_cases
from tests.plugins.sglang import cases as sglang_cases
from tests.plugins.vllm import cases as vllm_cases

# the gemm@1.0 kernel the CPU tests serve: out = a @ b.T
GEMM_KERNEL_SOURCE = "def run(*, a, b, out, **_):\n    out.copy_(a @ b.mT)\n"
# the same in hip, bf16 only
HIP_GEMM_KERNEL_SOURCE = """\
// out = a @ b.T in bf16, a thread an output element.
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

void run(const at::Tensor &a, const at::Tensor &b, bool trans_b,
         const std::optional<at::Tensor> &bias, at::ScalarType out_dtype, at::Tensor &out) {
    gemm<<<a.size(0), b.size(0), 0, at::hip::getCurrentHIPStream()>>>(
        static_cast<const __hip_bfloat16 *>(a.data_ptr()),
        static_cast<const __hip_bfloat16 *>(b.data_ptr()),
        static_cast<__hip_bfloat16 *>(out.data_ptr()), static_cast<int>(a.size(1)));
}

} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, bool trans_b, Tensor? bias, ScalarType out_dtype, "
          "Tensor(a!) out) -> ()",
          &run);
}
"""
# the same as a code object's device function (hsaco_gemm_code_object), and the hsaco kernel
# launching it from codeobjects.py (hsaco_gemm_sources)
HSACO_GEMM_DEVICE_SOURCE = """\
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
HSACO_GEMM_KERNEL_SOURCE = '''\
"""out = a @ b.T in bf16 from a code object, a thread an output element."""

import struct

from aklr.core.jit import hsaco

from .codeobjects import GEMM

_GEMM = hsaco.Kernel(GEMM, "gemm")


def run(*, a, b, trans_b, bias, out_dtype, out):
    kernarg = struct.pack("<QQQii", a.data_ptr(), b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])
    _GEMM.launch((a.shape[0],), (b.shape[0],), kernarg, device=a.device)
'''
KERNEL_TARGET = {"arch": "gfx950", "hardware": "MI355X"}  # as the committed schemas


def hsaco_gemm_code_object() -> bytes:
    """HSACO_GEMM_DEVICE_SOURCE's code object."""
    return hsaco_code_object(HSACO_GEMM_DEVICE_SOURCE)


@functools.cache
def hsaco_code_object(device_source: str) -> bytes:
    """``device_source`` built by ROCm's hipcc into a code object for the GPU's arch."""
    from torch.utils import cpp_extension

    arch = torch.cuda.get_device_properties(0).gcnArchName.partition(":")[0]
    with tempfile.TemporaryDirectory() as tmp:
        source, code_object = Path(tmp, "kernel.hip"), Path(tmp, "kernel.hsaco")
        source.write_text(device_source)
        hipcc = Path(cpp_extension.ROCM_HOME, "bin", "hipcc")
        subprocess.run(
            [hipcc, "--genco", f"--offload-arch={arch}", "-O3", source, "-o", code_object],
            check=True,
            capture_output=True,
        )
        return code_object.read_bytes()


def hsaco_gemm_sources() -> dict[str, str]:
    """codeobjects.py, holding hsaco_gemm_code_object as GEMM, for HSACO_GEMM_KERNEL_SOURCE."""
    encoded = base64.b64encode(hsaco_gemm_code_object()).decode()
    return {"codeobjects.py": f'"""The gemm."""\n\nimport base64\n\nGEMM = base64.b64decode("{encoded}")\n'}


@contextlib.contextmanager
def _one_rank_group(is_needed: bool):
    """A torch.distributed group of this process alone while the block runs, as validation puts a ranks
    contract's one rank in, where ``is_needed`` and no group is up yet."""
    import torch.distributed as dist

    if not is_needed or dist.is_initialized():
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{tmp}/rendezvous",
            rank=0,
            world_size=1,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


_NO_OP_KERNEL_SOURCE = "def run(**call): ...\n"  # a schema's kernel no test runs

# every contract's small Points (tests/ops/cases.py, tests/plugins/<framework>/cases.py),
# contract order: each contract's base Point first
_SMALL_POINTS: tuple[PointSpec, ...] = tuple(
    sorted(
        (
            *ops_cases.SMALL_POINTS,
            *sglang_cases.SMALL_POINTS,
            *vllm_cases.SMALL_POINTS,
        ),
        key=lambda point: point.contract_id,
    )
)
_GFX950_CONTRACT_IDS = (
    ops_cases.GFX950_CONTRACT_IDS | sglang_cases.GFX950_CONTRACT_IDS | vllm_cases.GFX950_CONTRACT_IDS
)


class OpCase:
    """One registered contract at its small Point."""

    contract: OpContract
    point: PointSpec

    def __init__(self, point: PointSpec) -> None:
        contract = ContractRegistry.default().get(point.contract_id)
        contract.check_point(point)
        if not contract.domain.contains(point):
            raise ValueError(f"{point.contract_id}: the Point is outside the domain")
        self.contract, self.point = contract, point

    @classmethod
    def all(cls) -> tuple[OpCase, ...]:
        """Every small Point's case, in contract order."""
        return tuple(cls(point) for point in _SMALL_POINTS)

    @classmethod
    def params(cls, points: Iterable[PointSpec]) -> list[Any]:
        """The cases of ``points`` in contract order as pytest params: the contract id, and ``gfx950``
        where the backend runs the Point only there."""
        cases = [cls(point) for point in sorted(points, key=lambda point: point.contract_id)]
        return [
            pytest.param(case, id=case.contract_id, marks=[pytest.mark.gfx950] if case.needs_gfx950 else [])
            for case in cases
        ]

    @classmethod
    def of(cls, contract_id: str) -> OpCase:
        """The first case of ``contract_id``; ``KeyError`` for a contract without one."""
        case = next((c for c in cls.all() if c.contract_id == contract_id), None)
        if case is None:
            raise KeyError(contract_id)
        return case

    @property
    def contract_id(self) -> str:
        """The contract id: the case's pytest id."""
        return self.contract.contract_id

    @property
    def needs_gfx950(self) -> bool:
        """Whether the backend runs the Point only on gfx950."""
        return (
            self.contract_id in _GFX950_CONTRACT_IDS
            or (
                self.contract_id,
                self.point.layout_variant_name,
            )
            in ops_cases.GFX950_LAYOUT_VARIANTS
        )

    def schema(
        self,
        kernel_source: str = _NO_OP_KERNEL_SOURCE,
        *,
        language: Language = "flydsl",
        other_sources: Mapping[str, str] | None = None,
    ) -> OpContract:
        """The schema serving the Point: the contract narrowed to one workload, its kernel
        ``kernel_source`` in kernel.py (kernel.hip in hip) beside ``other_sources``."""
        entry_file = f"kernel{SOURCE_SUFFIXES[language][0]}"
        sources = {entry_file: kernel_source, **(other_sources or {})}
        return self.contract.narrow(
            self.point,
            workloads=[self.contract.workload_for(self.point, {})],
            kernel={
                "language": language,
                "targets": [KERNEL_TARGET],
                "entry_point": f"{entry_file}::run",
                "sources": [{"path": p, "content": c} for p, c in sorted(sources.items())],
            },
        )

    @cached_property
    def request(self) -> TrialRequest:
        """``TrialRequest.of_workload`` of ``schema()``'s one workload, no thresholds, no
        incumbent, no baseline."""
        schema = self.schema()
        return TrialRequest.of_workload(
            str(op_dir(self.contract.op_package) / schema.relative_path()),
            schema,
            self.contract,
            schema.workloads[0],
            thresholds={},
            incumbent=None,
        )

    def inputs(self, input_set: str = "numeric/0") -> dict[str, Any]:
        """The row-major inputs of ``input_set`` drawn on the CPU (``row_major_inputs``)."""
        request = self.request
        return semantics.row_major_inputs(
            request.schema_json,
            request.workload_json,
            request.distributions,
            input_set=input_set,
            device="cpu",
        )

    def call(self, row_major_args: Mapping[str, Any], device: str = "cpu") -> CallArgs:
        """The api call of ``row_major_args`` on ``device`` (``TrialRequest.assemble_call``)."""
        return self.request.assemble_call(row_major_args, device)

    def reference_outputs(self, row_major_args: Mapping[str, Any]) -> dict[str, Any]:
        """The op's reference of ``row_major_args``, by name: one rank's call."""
        # shortcut: one rank's inputs only, so no group of more; draw every rank's (rank= of
        # row_major_inputs) when a ranks contract gets a Small Point
        if (n_ranks := self.contract.ranks_at(self.point)) > 1:
            raise NotImplementedError(f"{self.contract.contract_id}: a group of {n_ranks} ranks")
        call = self.request.reference_call(row_major_args, "cpu")
        if self.contract.ranks is None:
            return dict(self.contract.reference.reference(**call))
        return dict(self.contract.reference.reference([call], 0))

    def compare(
        self,
        actual_outputs: Mapping[str, Any],
        row_major_args: Mapping[str, Any],
    ) -> dict[str, str]:
        """The reference's check of ``actual_outputs`` against its outputs of ``row_major_args``, on the CPU."""
        return self.contract.reference.check(
            {o: t.cpu() for o, t in actual_outputs.items()},
            {o: t.cpu() for o, t in self.reference_outputs(row_major_args).items()},
            self.request.schema,
        )

    def check_backend_matches_reference(self) -> None:
        """The backend, an implementation independent of the reference, checks the reference: equal
        integer outputs, floating ones at an SQNR, the contract's own min_sqnr_db where it
        declares one. Other bounds are the kernels', per element; the backend is held to 20 dB."""
        args = self.inputs()
        call = self.call(args, "cuda")
        baseline = self.contract.baseline
        with _one_rank_group(isinstance(baseline, DistOpBaseline)):
            baseline.setup()
            try:
                outputs = {**baseline.baseline(**call), **{n: call[n] for n in self.contract.mutable_inputs}}
            finally:
                baseline.teardown()
        results = self.compare(outputs, args)
        schema = self.request.schema
        integral = {
            o for o in schema.mutable_tensors if semantics.DTYPE_TRAITS[self.point.dtypes[o]].is_integral
        }
        assert {o: why for o, why in results.items() if o in integral} == {}
        reference = {o: t.float() for o, t in self.reference_outputs(args).items()}
        tensors = schema.tensors()
        for o in set(schema.mutable_tensors) - integral:
            bound = tensors[o].error_bounds[0][1]
            # shortcut: 20 dB uncalibrated (MI300X skips), set from the gfx950 SQNRs once measured
            min_sqnr_db = bound.min_sqnr_db if isinstance(bound, semantics.SqnrBound) else 20.0
            error = (outputs[o].cpu().float() - reference[o]).norm()
            assert error <= 10 ** (-min_sqnr_db / 20) * reference[o].norm(), (
                o,
                min_sqnr_db,
                (20 * (reference[o].norm() / error).log10()).item(),
            )
