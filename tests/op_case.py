# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

import aklr.ops
from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch import KernelDispatcher
from aklr.core.schema import (
    Build,
    Evaluation,
    Kernel,
    Schema,
    SchemaCollection,
    SourceFile,
    Workload,
)
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.ops.fused_moe import fused_moe

GEMM_KERNEL = "def run(a, b, bias, out):\n    out.copy_(a @ b.mT)\n"

_FUSED_MOE = {
    "n_tokens": 4,
    "model_dim": 512,
    "n_experts": 8,
    "topk": 2,
    "inter_dim": 256,
    "global_experts": None,
}
_GEMM_FP8 = {"M": 4, "N": 256, "K": 256}

_CASES: tuple[tuple[Callable[..., Any], str, dict[str, Any]], ...] = (
    (aklr.ops.add_rms_norm, "add_rms_norm", {"n_tokens": 4, "hidden_size": 6144}),
    (aklr.ops.gemm_bf16_nt, "gemm_bf16_nt", {"M": 4, "N": 32, "K": 64}),
    (aklr.ops.gemm_bf16_nt, "gemm_bf16_nt_shuffled", {"M": 4, "N": 64, "K": 64}),
    (aklr.ops.gemm_fp8_nt_blockscaled, "gemm_fp8_nt_blockscaled", _GEMM_FP8),
    (aklr.ops.gemm_fp8_nt_blockscaled, "gemm_fp8_nt_blockscaled_mn", _GEMM_FP8),
    (
        aklr.ops.gemm_fp8_nt_blockscaled,
        "gemm_fp8_nt_blockscaled_shuffled_mn",
        _GEMM_FP8,
    ),
    (
        aklr.ops.gemm_fp8_nt_blockscaled,
        "gemm_fp8_nt_blockscaled_shuffled_k",
        _GEMM_FP8,
    ),
    (
        aklr.ops.mhc_fused_post_pre,
        "mhc_fused_post_pre",
        {"n_tokens": 4, "hidden_size": 4096},
    ),
    (
        aklr.ops.fp8_paged_mqa_logits,
        "fp8_paged_mqa_logits",
        {
            "batch": 2,
            "n_heads": 64,
            "n_pages": 4,
            "page_tokens": 64,
            "max_blocks": 2,
            "max_seq_len": 128,
        },
    ),
    (
        aklr.ops.flash_mla_with_kvcache,
        "flash_mla_with_kvcache_sparse",
        {"batch": 1, "n_heads": 64, "n_pages": 2, "page_size": 256, "n_indices": 128},
    ),
    (
        aklr.ops.flash_mla_with_kvcache,
        "flash_mla_with_kvcache_dense",
        {"batch": 2, "n_heads": 96, "n_pages": 64, "page_size": 1, "table_width": 16},
    ),
    (fused_moe, "fused_moe_shuffled_16x16", _FUSED_MOE),
    (fused_moe, "fused_moe_gate_up_interleaved", _FUSED_MOE),
    (fused_moe, "fused_moe_situv2_gate_up_interleaved", _FUSED_MOE),
)


def dispatcher(*schemas: Schema) -> KernelDispatcher:
    return KernelDispatcher(SchemaCollection(schemas={s.name: [s] for s in schemas}))


@dataclass(frozen=True)
class OpCase:
    api: Callable[..., Any]
    template: Schema
    axes: Mapping[str, Any]

    @classmethod
    def all(cls) -> tuple[OpCase, ...]:
        return tuple(
            cls(api, next(s for s in api.schemas if s.schema_name({}) == id_), axes)
            for api, id_, axes in _CASES
        )

    @classmethod
    def of(cls, id_: str) -> OpCase:
        return next(case for case in cls.all() if case.id == id_)

    @property
    def id(self) -> str:
        return self.template.schema_name({})

    def call(self, device: str = "cpu", seed: int = 0) -> dict[str, Any]:
        init = self.template.source_to_function("init")
        return init(**self.axes, device=device, seed=seed)

    def reference(self, call: Mapping[str, Any]) -> list[torch.Tensor]:
        reference = self.template.source_to_function("reference")
        params = inspect.signature(reference).parameters
        with torch.no_grad():
            result = reference(**{n: v for n, v in call.items() if n in params})
        return list(result) if isinstance(result, (tuple, list)) else [result]

    @property
    def output_dtypes(self) -> list[torch.dtype]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.schema().outputs.values()]

    def schema(self, kernel: Kernel | None = None, workloads: Sequence[Workload] = ()) -> Schema:
        return replace(
            self.template.for_call(self.call()),
            kernel=kernel,
            workloads=list(workloads),
        )

    def served(
        self,
        source: str,
        *,
        dps: bool = True,
        evaluations: Sequence[Evaluation] | None = None,
    ) -> Schema:
        target = DeviceTarget.current()
        if evaluations is None:
            evaluations = [Evaluation(hardware=str(target), accuracy="passed")]
        kernel = Kernel(
            name=f"{self.id}_{'dps' if dps else 'returning'}",
            author="tests",
            build=Build(
                language="triton",
                target_hardware=[target.arch],
                entry_point="kernel.py::run",
                destination_passing_style=dps,
            ),
            sources=[SourceFile(path="kernel.py", content=source)],
        )
        var_axes = {a: v for a, v in self.axes.items() if a in self.template.var_axes}
        return self.schema(kernel, [Workload(axes=var_axes, evaluations=evaluations)])
