# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA decode query padded to the head count aiter's decode kernels take, as vLLM pads it on ROCm
(``AiterMLAHelper.get_mla_padded_q``, ``v1/attention/backends/mla/rocm_aiter_mla.py``).

aiter's MLA decode kernels exist at their native head counts only. Where a rank's decode query holds no divisor of
one (Kimi-K3's 96 gathered heads under decode context parallelism, for the 128-head kernel), vLLM tiles the heads
(``q.repeat(1, reps, 1)[:, :m].contiguous()``, two launches), runs the kernel, and reads the first ``n_heads`` heads
of its output. The op is that tiling, the query's bytes as they are."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, FiniteDistribution, ToleranceBound
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.vllm.layers.mla_query_heads.baseline import MlaQueryHeadsBaseline
from aklr.plugins.vllm.layers.mla_query_heads.reference import MlaQueryHeadsReference

__all__ = ["CONTRACTS", "MLA_QUERY_HEADS"]

MLA_QUERY_HEADS = OpContract(
    contract_id="mla_query_heads@1.0",
    description=(
        "out[t, h, :] = q[t, h mod n_heads, :] for every h < out_heads: the query's heads, then its first ones again, "
        "out_heads heads in all, the bytes unchanged (vLLM's q.repeat(1, reps, 1)[:, :out_heads])."
    ),
    op_type="mla_query_heads",
    axes={
        "n_tokens": axis("Query rows of the step.", 1, 16384, is_var=True),
        "n_heads": axis("Heads of the query.", 1, 1024),
        "head_dim": axis(
            "Width of a head's row (vLLM's kv_lora_rank + qk_rope_head_dim).", 16, 4096, multiple_of=16
        ),
        "n_out_heads": derived_axis("Heads of the output.", "out_heads"),
    },
    attrs={
        "out_heads": attr(
            "The head count the decode kernel takes.",
            128,
            16,
            32,
            64,
            256,
            default=128,
            param_name="out_heads",
            abbrev="o",
        ),
    },
    # more heads than the query's; a kernel indexes the output's bytes in 32 bits
    constraints=("out_heads > n_heads", "n_tokens * out_heads * head_dim < 2147483648"),
    inputs={
        "q": TensorSpec(
            description="The decode query, one byte an element (vLLM's static fp8 query under an fp8 KV cache).",
            shape=("n_tokens", "n_heads", "head_dim"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=FiniteDistribution(),
        ),
    },
    outputs={
        "out": TensorSpec(
            description="The query with its heads tiled to out_heads.",
            shape=("n_tokens", "n_out_heads", "head_dim"),
            dtype=DTypeSpec(same_as_tensor="q", param_name="out_dtype"),
            error_bounds=ToleranceBound(absolute_tolerance=0.0, relative_tolerance=0.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
    reference=MlaQueryHeadsReference(),
    baseline=MlaQueryHeadsBaseline(),
)

CONTRACTS = (MLA_QUERY_HEADS,)
