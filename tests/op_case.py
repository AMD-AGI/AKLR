# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any

import torch

from aklr.core.contract import semantics
from aklr.core.contract.fastkey import CallArgs
from aklr.core.contract.locate import op_dir, op_module
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import CompareResult
from aklr.core.contract.spec import SOURCE_SUFFIXES, Language, PointSpec
from aklr.core.validation.stages._bench._evaluation._trials import TrialRequest
from tests.ops import cases as ops_cases
from tests.plugins.sglang import cases as sglang_cases

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
KERNEL_TARGET = {"arch": "gfx950", "hardware": "MI355X"}  # as the committed schemas
_NO_OP_KERNEL_SOURCE = "def run(**call): ...\n"  # a schema's kernel no test runs

# every contract's small Points (tests/ops/cases.py, tests/plugins/<framework>/cases.py),
# contract order: each contract's base Point first
_SMALL_POINTS: tuple[PointSpec, ...] = tuple(
    sorted(
        (*ops_cases.SMALL_POINTS, *sglang_cases.SMALL_POINTS),
        key=lambda point: point.contract_id,
    )
)
_GFX950_CONTRACT_IDS = ops_cases.GFX950_CONTRACT_IDS | sglang_cases.GFX950_CONTRACT_IDS


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
        return self.contract_id in _GFX950_CONTRACT_IDS

    def schema(
        self,
        kernel_source: str = _NO_OP_KERNEL_SOURCE,
        *,
        language: Language = "flydsl",
        other_sources: Mapping[str, str] | None = None,
        state_bytes: int = 0,
        peer_bytes: int = 0,
    ) -> OpContract:
        """The schema serving the Point: the contract narrowed to one workload, its kernel
        ``kernel_source`` in kernel.py (kernel.hip in hip) beside ``other_sources``, keeping
        ``state_bytes`` between calls and exchanging through ``peer_bytes``."""
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
                "state_bytes": state_bytes,
                "peer_bytes": peer_bytes,
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
            baseline=None,
        )

    def inputs(self, input_set: str = "numeric/0") -> dict[str, Any]:
        """The row-major inputs of ``input_set`` on the CPU (``row_major_inputs``)."""
        request = self.request
        return semantics.row_major_inputs(
            request.schema_json,
            request.workload_json,
            request.distributions,
            input_set=input_set,
        )

    def call(self, row_major_args: Mapping[str, Any], device: str = "cpu") -> CallArgs:
        """The api call of ``row_major_args`` on ``device`` (``TrialRequest.assemble_call``)."""
        return self.request.assemble_call(row_major_args, device)

    def reference_outputs(self, row_major_args: Mapping[str, Any]) -> dict[str, Any]:
        """The op's float64 reference of ``row_major_args``, by name in ``contract.mutable_tensors``."""
        reference_module = op_module(self.contract.op_package, "reference")
        reference_fn = getattr(reference_module, self.contract.reference_name)
        return self.contract.mutable_tensors_by_name(
            reference_fn(self.point, row_major_args, semantics.to_float64)
        )

    def compare(
        self,
        actual_outputs: Mapping[str, Any],
        row_major_args: Mapping[str, Any],
        *,
        baseline_outputs: Mapping[str, Any] | None = None,
    ) -> dict[str, CompareResult]:
        """``compare_outputs`` of ``actual_outputs`` against the reference of ``row_major_args``, on the CPU."""
        compare_kwargs = {"reference_outputs": self.reference_outputs(row_major_args)}
        if self.request.has_error_bound("dot_product"):
            magnitudes = {
                n: semantics.absolute_value(v, self.point.dtypes[n])
                for n, v in row_major_args.items()
                if torch.is_tensor(v)
            }
            compare_kwargs["magnitude_outputs"] = self.reference_outputs({**row_major_args, **magnitudes})
        if baseline_outputs is not None:
            compare_kwargs["baseline_outputs"] = baseline_outputs
        compare_kwargs = {
            kind: {o: t.cpu() for o, t in tensors.items()} for kind, tensors in compare_kwargs.items()
        }
        return semantics.compare_outputs(
            self.request.schema_json,
            {o: t.cpu() for o, t in actual_outputs.items()},
            **compare_kwargs,
        )
