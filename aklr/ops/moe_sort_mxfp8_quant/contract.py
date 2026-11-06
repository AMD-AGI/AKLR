# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The front of aiter's MXFP8-activation MoE: sort the routed pairs into expert blocks, zero
the stage-2 accumulator and quantize the activations to MXFP8 with the scales in sorted rows."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    OutputRoundingBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "MOE_SORT_MXFP8_QUANT"]

_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
# a buffer's earlier contents: any int32
_ANY_INT32 = IntRangeDistribution(lo=-(2**31), hi=2**31 - 1)

MOE_SORT_MXFP8_QUANT = OpContract(
    contract_id="moe_sort_mxfp8_quant@1.0",
    description=(
        "What aiter's fused_moe runs before stage 1 for MXFP8 activations (a8w4/a8w8, "
        "per_1x32, no expert parallelism) while stage 1 takes the fused sort-and-quant: the "
        "moe sorting and fused_dynamic_mxfp8_quant_moe_sort. The n_tokens * topk routed pairs "
        "(token t, slot k, expert topk_ids[t, k]) are ordered by expert, then by t * topk + k, "
        "into blocks of block_size rows, each expert's rows in whole blocks from row 0 in "
        "increasing expert order; a row holds (k << 24) | t and the pair's weight, a block's "
        "unused rows (topk << 24) | n_tokens and weight 0. sorted_expert_ids holds each "
        "block's expert, num_valid_ids [rows of the blocks, n_tokens]; moe_buf is zeroed. "
        "Each token row is quantized to E4M3FN in groups of 32 with an E8M0 scale "
        "2**ceil(log2(max(amax, 1e-10) * (1/448))), its exponent byte written for each sorted "
        "row of a pair at aiter's mx_scale_shuffle_idx address (0 for a pair of weight 0); a "
        "token whose weights are all 0 gets a zero row. Rows past the blocks, a block's unused "
        "rows' scales and the scale padding columns keep their contents."
    ),
    op_type="moe_sort_mxfp8_quant",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 2048, is_var=True),
        "n_sorted_rows": axis(
            "Rows of the sorted buffers (aiter allocates n_tokens * topk + n_experts * "
            "block_size - topk, rounded up to whole blocks).",
            32,
            2**24,
            multiple_of=32,
            is_var=True,
            any_size=True,
        ),
        "n_blocks": axis(
            "Entries of sorted_expert_ids: n_sorted_rows // block_size.", 1, 2**24, is_var=True, any_size=True
        ),
        "topk": axis("Experts per token.", 1, 32),
        "hidden_size": axis("Elements per token row, divisible by 32.", 32, 8192, multiple_of=32),
        "n_groups": derived_axis("MXFP8 groups of 32 elements per row.", "hidden_size // 32"),
        "scale_cols": derived_axis(
            "Bytes per sorted row in sorted_scales: n_groups rounded up to 8.",
            "(hidden_size // 32 + 7) // 8 * 8",
        ),
    },
    attrs={
        # no tensor carries it; DeepSeek-V4.1-Flash: 384 routed + 1 shared, its DSpark draft 128 + 1
        "n_experts": attr(
            "Experts the routed ids index, a fused shared expert included.", 385, 129, param_name="n_experts"
        ),
        "block_size": attr(
            "Rows per sorted block: the M tile of the expert GEMMs.", 32, default=32, param_name="block_size"
        ),
    },
    constraints=(
        # aiter quantizes stage 1 with the fused sort-and-quant kernel up to 8 * 256 / topk
        # tokens; past it, a split path that keeps zero-weight pairs' scales
        "n_tokens * topk <= 2048",
        "n_sorted_rows >= n_tokens * topk + n_experts * block_size - topk",
        "n_blocks * block_size == n_sorted_rows",
    ),
    inputs={
        "hidden_states": TensorSpec(
            description="Token activations.",
            shape=("n_tokens", "hidden_size"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=_FINITE,
        ),
        "topk_ids": TensorSpec(
            # aiter's opus sort keeps one pair of an expert repeated in a row
            description="Chosen experts per token, distinct within a row.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_experts - 1", distinct=True),
        ),
        "topk_weights": TensorSpec(
            description="Routing weight of each chosen expert.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=_FINITE,
        ),
        "sorted_ids": TensorSpec(
            description="Per sorted row: (slot << 24) | token, or (topk << 24) | n_tokens on an unused row.",
            shape=("n_sorted_rows",),
            dtype=_INT32,
            distribution=_ANY_INT32,
            mutable=True,
        ),
        "sorted_weights": TensorSpec(
            description="Per sorted row: the pair's weight, 0 on an unused row.",
            shape=("n_sorted_rows",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=_FINITE,
            mutable=True,
            error_bounds=OutputRoundingBound(),
        ),
        "sorted_expert_ids": TensorSpec(
            description="Per block: its expert.",
            shape=("n_blocks",),
            dtype=_INT32,
            distribution=_ANY_INT32,
            mutable=True,
        ),
        "sorted_scales": TensorSpec(
            description=(
                "E8M0 exponent bytes of each sorted row's groups, 32-row tiles of "
                "32 * scale_cols bytes; row r, group g at byte (r // 32) * 32 * scale_cols + "
                "(g // 8) * 256 + (g % 4) * 64 + (r % 16) * 4 + (g % 8) // 4 * 2 + (r % 32) // 16."
            ),
            shape=("n_sorted_rows", "scale_cols"),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            distribution=_FINITE,
            mutable=True,
        ),
    },
    outputs={
        "num_valid_ids": TensorSpec(
            description="[rows of the experts' blocks, n_tokens].",
            shape=(2,),
            dtype=_INT32,
        ),
        "moe_buf": TensorSpec(
            description="The stage-2 accumulator, zeroed.",
            shape=("n_tokens", "hidden_size"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            error_bounds=OutputRoundingBound(),
        ),
        "quantized": TensorSpec(
            description="E4M3FN activations per token, zero for a token whose weights are all 0.",
            shape=("n_tokens", "hidden_size"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            error_bounds=OutputRoundingBound(),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (MOE_SORT_MXFP8_QUANT,)
