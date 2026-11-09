# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    BaselineRelativeBound,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "DEEPSEEK_V4_ATTENTION_Q_PREP"]

# DeepSeek-V4's MQA head: 448 no-RoPE dims, then 64 RoPE dims
_HEAD_DIM, _ROPE_DIM = 512, 64

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))

_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))

_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))

_EPS = ScalarSpec(
    description="RMSNorm epsilon.",
    dtype=DType.FLOAT32,
    interval=(0.0, 1e-5),
    # DeepSeek-V4's rms_norm_eps, 1e-20, as float32
    default=9.99999968265522e-21,
)

DEEPSEEK_V4_ATTENTION_Q_PREP = OpContract(
    contract_id="deepseek_v4_attention_q_prep@1.0",
    description=(
        "What sglang's MQALayer._forward_prepare runs on ROCm gfx950 between wqkv_a and attention "
        "when the K store also ropes the queries: the q_lora_rank columns of qkv_a are RMS-normalized "
        "with q_norm_weight, rounded to BF16 and quantized per 32 columns to E4M3FN with a UE8M0 scale "
        "2**ceil(log2(max(amax, 1e-10) * float32(1 / 448))) (q_lora_quant, q_lora_scale); q is that "
        "operand times wq_b (E4M3FN, one UE8M0 scale per 32 x 32 block) transposed, rounded to BF16, "
        "whose last 64 dims of each head take RoPE: pair (2j, 2j + 1) times (cos, sin) = "
        "freqs[positions, 2j], freqs[positions, 2j + 1]. The last 512 columns of qkv_a (the KV row) are "
        "RMS-normalized with kv_norm_weight; dims 0-447 are quantized per 64 to E4M3FN with a UE8M0 "
        "scale 2**ceil(log2(max(amax, 1e-4) / 448)), dims 448-511 take the same RoPE in BF16, and the "
        "row is written at slot swa_loc of kv_cache (sglang's V4 layout: page p's slot s holds its 448 "
        "codes then its 64 BF16 values at byte 576 * s of row p, and its 7 scale bytes at byte "
        "576 * page_size + 8 * s); a negative slot writes nothing. All other bytes of kv_cache are "
        "preserved."
    ),
    op_type="deepseek_v4",
    axes={
        "n_tokens": axis("Tokens in this call, padding rows included.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "q_lora_rank": axis("Width of the query LoRA rank, wq_b's input.", 32, 8192, multiple_of=32),
        "n_pages": axis("Pages of kv_cache.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_positions": axis("Rows of the cos/sin table.", 1, 2**25, is_var=True, any_size=True),
        "qkv_dim": derived_axis(
            "Columns of qkv_a: the query LoRA rank, then the KV row.", "q_lora_rank + 512"
        ),
        "q_dim": derived_axis("Output columns of wq_b.", "n_heads * 512"),
        "q_lora_groups": derived_axis(
            "Scales of a quantized q_lora row: one per 32 columns.", "q_lora_rank // 32"
        ),
        "wq_b_scale_rows": derived_axis("Rows of wq_b's scales: one per 32 output columns.", "n_heads * 16"),
        "page_bytes": derived_axis(
            "Bytes of a kv_cache page: 584 a slot, padded to a multiple of 576.",
            "(page_size * 584 + 575) // 576 * 576",
        ),
    },
    attrs={
        # sglang's SWA pool on ROCm: 256 (DeepSeekV4TokenToKVPool.swa_page_size)
        "page_size": attr("Slots in a page of kv_cache.", 256, 64, 128, param_name="page_size"),
    },
    constraints=(
        # every row writes its own slot
        "n_pages * page_size >= n_tokens",
    ),
    inputs={
        "qkv_a": TensorSpec(
            description="wqkv_a's output: the query LoRA rank's columns, then the KV row.",
            shape=("n_tokens", "qkv_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "q_norm_weight": TensorSpec(
            description="q_norm's weight.",
            shape=("q_lora_rank",),
            dtype=_BF16,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "kv_norm_weight": TensorSpec(
            description="kv_norm's weight.",
            shape=(_HEAD_DIM,),
            dtype=_BF16,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "wq_b": TensorSpec(
            description="wq_b's weight, output columns by input columns.",
            shape=("q_dim", "q_lora_rank"),
            dtype=_FP8,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "wq_b_scale": TensorSpec(
            description="wq_b's scales, one per 32 x 32 block.",
            shape=("wq_b_scale_rows", "q_lora_groups"),
            dtype=_E8M0,
            distribution=LogUniformDistribution(log2_lo="-8", log2_hi="8"),
            lifetime="weight",
        ),
        "freqs": TensorSpec(
            description="Per position, the 32 RoPE pairs' (cos, sin), interleaved.",
            shape=("n_positions", _ROPE_DIM),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "positions": TensorSpec(
            description="Each token's position, a row of freqs.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
        "swa_loc": TensorSpec(
            description=(
                "Each token's kv_cache slot, page * page_size + slot; negative: not written. Rows "
                "sharing a slot leave one of their rows there."
            ),
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=0, hi="n_pages * page_size - 1", distinct=True),
        ),
        "kv_cache": TensorSpec(
            description="This layer's SWA pool in sglang's V4 layout, a page a row.",
            shape=("n_pages", "page_bytes"),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            distribution=FiniteDistribution(),
            mutable=True,
        ),
        "q_eps": _EPS,
        "kv_eps": _EPS,
    },
    outputs={
        "q": TensorSpec(
            description="The queries, the last 64 dims of each head roped.",
            shape=("n_tokens", "n_heads", _HEAD_DIM),
            dtype=_BF16,
            # sglang's own GEMM sums in float32 and rounds to BF16 before the RoPE, which can
            # cancel: a pair one BF16 step off moves a small output by a step of its larger input.
            # wq_b's block scales spread the outputs' magnitudes over 2**16, so no single
            # tolerance fits; the baseline's error does
            error_bounds=BaselineRelativeBound(),
        ),
        "q_lora_quant": TensorSpec(
            description="The normalized q_lora on the E4M3FN grid, what wq_b and the indexer read.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_FP8,
            # a mean square summed in another order (sglang's Triton tree, the reference's exact
            # sum) can move a normalized BF16 value by an ulp: across an E4M3 midpoint, or, at a
            # group's amax on 448 times a power of two, onto the next scale, which halves or
            # doubles the group's codes (MI355X bench: one group of 7680 in 2 of 52 workloads)
            error_bounds=ToleranceBound(absolute_tolerance=2**-9, relative_tolerance=1.0),
        ),
        "q_lora_scale": TensorSpec(
            description="q_lora_quant's scales, one per 32 columns.",
            shape=("n_tokens", "q_lora_groups"),
            dtype=_E8M0,
            # the same: one exponent step (with UE8M0's unit roundoff, 0.5, a factor of 2)
            error_bounds=ToleranceBound(absolute_tolerance=0.0, relative_tolerance=0.5),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # kv_cache is integer: exact, so no violation
    compare=CompareSpec(determinism="bitwise"),
)

CONTRACTS = (DEEPSEEK_V4_ATTENTION_Q_PREP,)
