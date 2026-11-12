# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

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
from aklr.core.validation import ExecutionPolicy
from aklr.core.validation.stages.bench import InitSampler, InputSampler, ReferenceCall
from aklr.core.validation.stages.workload_key import WorkloadKey
from aklr.ops.topk import biased_grouped_topk
from aklr.ops.fused_moe import fused_moe
from aklr.ops.mega_moe import mega_moe

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
    (aklr.ops.rms_norm, "rms_norm", {"n_tokens": 4, "hidden_size": 6144}),
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
    (
        aklr.ops.flash_mla_with_kvcache,
        "flash_mla_with_kvcache_ragged",
        {"batch": 2, "query_length": 1, "n_heads": 96, "n_pages": 64, "page_size": 1, "table_width": 16},
    ),
    *(
        (
            aklr.ops.paged_attention_decode,
            name,
            {"batch": 2, "n_heads": 16, "n_kv_heads": 2, "head_dim": 64, "n_blocks": 8, "block_size": 16}
            | {"max_blocks": 4},
        )
        for name in ("paged_attention_decode", "paged_attention_decode_kv_split_pages")
    ),
    # vLLM's verify step of 8 new tokens on its split cache; a windowed one past its window
    (
        aklr.ops.unified_attention,
        "unified_attention_kv_split_pages",
        {"total_q": 12, "batch": 2, "n_heads": 16, "n_kv_heads": 2, "head_dim": 64, "n_blocks": 4}
        | {"block_size": 16, "max_blocks": 4, "max_seqlen_q": 8, "sliding_window": 0},
    ),
    (
        aklr.ops.unified_attention,
        "unified_attention",
        {"total_q": 5, "batch": 2, "n_heads": 8, "n_kv_heads": 1, "head_dim": 128, "n_blocks": 96}
        | {"block_size": 32, "max_blocks": 96, "max_seqlen_q": 4, "sliding_window": 2048},
    ),
    (aklr.ops.bmm_bf16_nt, "bmm_bf16_nt", {"batch_size": 3, "M": 4, "N": 32, "K": 64}),
    (aklr.ops.bmm_bf16_nn, "bmm_bf16_nn", {"batch_size": 3, "M": 4, "N": 32, "K": 64}),
    (
        biased_grouped_topk,
        "biased_grouped_topk",
        {"n_tokens": 4, "n_experts": 64, "topk": 6, "num_expert_group": 8, "topk_group": 4},
    ),
    (aklr.ops.per_group_quant_fp8, "per_group_quant_fp8", {"n_tokens": 4, "hidden_size": 256}),
    (aklr.ops.per_group_quant_mxfp8, "per_group_quant_mxfp8", {"n_tokens": 4, "hidden_size": 256}),
    (
        aklr.ops.sparse_prefill_mxfp8,
        "sparse_prefill_mxfp8",
        {
            "n_tokens": 4,
            "n_heads": 16,
            "prefix_rows": 64,
            "extend_rows": 8,
            "prefix_nnz": 32,
            "extend_nnz": 8,
            "n_offsets": 5,
        },
    ),
    (fused_moe, "fused_moe_shuffled_16x16", _FUSED_MOE),
    (fused_moe, "fused_moe_gate_up_interleaved", _FUSED_MOE),
    (fused_moe, "fused_moe_situv2_gate_up_interleaved", _FUSED_MOE),
    (fused_moe, "fused_moe_swiglu_shuffled_16x16", _FUSED_MOE),
    (
        mega_moe,
        "mega_moe_gate_up_interleaved",
        {"n_tokens": 4, "model_dim": 512, "n_experts": 2, "topk": 3, "inter_dim": 256, "ep_size": 2},
    ),
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
    def from_id(cls, id_: str) -> OpCase:
        return next(case for case in cls.all() if case.id == id_)

    @property
    def id(self) -> str:
        return self.template.schema_name({})

    @property
    def ranks(self) -> int:
        """How many ranks call the op together, as the bench runs it."""
        return ExecutionPolicy.for_schema(self.schema()).ranks_per_task

    @property
    def key(self) -> WorkloadKey:
        """Its workload's key: the case's Var axes."""
        return WorkloadKey({a: v for a, v in self.axes.items() if a in self.template.var_axes})

    def call(self, device: str = "cpu", trial: int = 0) -> dict[str, Any]:
        """Rank 0's call of trial, as the bench draws it for this workload: a single rank's too."""
        return InputSampler.for_workload(self.schema(), self.key).draw(trial, 0, device)

    def calls(self, device: str = "cpu", trial: int = 0) -> list[dict[str, Any]]:
        """Every rank's call of trial, as the bench draws them."""
        return InputSampler.for_workload(self.schema(), self.key).draw_group(trial, self.ranks, device)

    def reference(self, calls: Sequence[Mapping[str, Any]], rank: int = 0) -> list[torch.Tensor]:
        """Rank rank's Reference on the group's calls, as the bench passes them."""
        with torch.no_grad():
            return ReferenceCall(self.schema()).outputs(calls, rank)

    @property
    def output_dtypes(self) -> list[torch.dtype]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.schema().outputs.values()]

    def schema(self, kernel: Kernel | None = None, workloads: Sequence[Workload] = ()) -> Schema:
        """The template at the case's axes, as for_call reads it off a call: one its init draws on the meta
        device, which has shapes and dtypes and no data."""
        return replace(
            self.template.for_call(InitSampler(self.template, dict(self.axes)).draw(0, 0, "meta")),
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
        return self.schema(kernel, [Workload(axes=self.key.axes, evaluations=evaluations)])
