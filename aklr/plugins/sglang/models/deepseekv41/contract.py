# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's calls on one rank of its TP group, where sglang serves them without a
monokernel: every prefill, and the decode and verify steps a monokernel does not take.

A model is a scenario: DeepSeek-V4.1-Flash at TP2 is 32 query heads on a rank."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    Deepseekv4Fp8KvCacheDistribution,
    Deepseekv4KvPages,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    AttrSpec,
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseekv41.baseline import (
    Deepseekv41Mxfp8LinearBaseline,
    Deepseekv41SparseMlaBaseline,
)
from aklr.plugins.sglang.models.deepseekv41.reference import (
    Deepseekv41Mxfp8LinearReference,
    Deepseekv41SparseMlaReference,
)

__all__ = ["CONTRACTS", "DEEPSEEKV41_MXFP8_LINEAR", "DEEPSEEKV41_SPARSE_MLA", "MXFP8_BLOCK", "ROPE_DIM"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FINITE = FiniteDistribution()
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# The RoPE part of a head: its last values, rotated in pairs
ROPE_DIM = 64
# Values sharing an MXFP8 scale: 32 along the reduction, and 32 weight rows
MXFP8_BLOCK = 32
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))
# Each pool is pages of page_size slots: 576-byte payloads, then 8-byte scale slots.
_PAGE_SIZES = "{0} == 2 or {0} == 64 or {0} == 128 or {0} == 256"
_SLOT_LIMIT = 2**31  # slots are int32 indices
_KV_POOL = DTypeSpec(any_of=(DType.UINT8, DType.FLOAT8_E4M3FN))


def _slots(description: str, width: str, pool: str, present_when: str | None = None) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("n_tokens", 1, width),
        dtype=_INT32,
        present_when=present_when,
        distribution=IntRangeDistribution(lo="-1", hi=f"{pool} - 1"),
    )


def _presence(tensor: str, abbrev: str) -> AttrSpec:
    return attr(f"Whether {tensor} is passed.", False, True, default=False, abbrev=abbrev)


DEEPSEEKV41_SPARSE_MLA = OpContract(
    contract_id="deepseekv41_sparse_mla@1.0",
    description=(
        "sglang's sparse MLA of a DeepSeek-V4.1 layer on gfx950 (aiter_sparse_decode_fwd): each "
        "head of each token attends the slots its indices name (-1: none) in the layer's KV pool "
        "and in the optional extra one, with scores q . row * sm_scale; sinks join the "
        "normalisation only; the output is the softmax-weighted sum of the rows, 0 for a token "
        "that reads no row. With rope_freqs, the last 64 values of every head are then rotated "
        "back in pairs (a, b) by the pair's (c, s) at the token's position: (a c + b s, b c - a s)."
    ),
    op_type="deepseekv41",
    axes={
        "n_tokens": axis("Tokens in this call, one query each.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "n_pages": axis("Pages of the KV pool.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
        "page_size": axis("Slots per page of the KV pool.", 2, 256),
        "n_indices": axis("Indices per token into the KV pool.", 1, 65536),
        "n_extra_pages": axis("Pages of the extra KV pool.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
        "extra_page_size": axis("Slots per page of the extra KV pool.", 2, 256),
        "n_extra_indices": axis("Indices per token into the extra KV pool.", 1, 65536),
        "n_positions": axis("Rows of the RoPE table.", 1, 2**21, is_var=True, any_size=True),
    },
    attrs={
        "has_extra_kv_cache": _presence("extra_kv_cache", "extra"),
        "has_extra_indices": _presence("extra_indices", "extra_idx"),
        "has_rope_freqs": _presence("rope_freqs", "inv_rope"),
        "has_positions": _presence("positions", "pos"),
    },
    constraints=(
        # the extra pool comes with its indices, the RoPE table with the tokens' positions
        "has_extra_kv_cache == has_extra_indices",
        "has_rope_freqs == has_positions",
        _PAGE_SIZES.format("page_size"),
        _PAGE_SIZES.format("extra_page_size"),
        f"n_pages * page_size <= {_SLOT_LIMIT}",
        f"n_extra_pages * extra_page_size <= {_SLOT_LIMIT}",
    ),
    inputs={
        "q": TensorSpec(
            description="Each token's query per head, the RoPE part last.",
            shape=("n_tokens", 1, "n_heads", 512),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "kv_cache": TensorSpec(
            description="The layer's KV pool: DeepSeek-V4's packed fp8 slots.",
            shape=("n_pages", "page_size", 1, 584),
            dtype=_KV_POOL,
            distribution=Deepseekv4Fp8KvCacheDistribution(),
            lifetime="weight",
        ),
        "indices": _slots(
            "Slots of the KV pool each token reads; negative: skipped.", "n_indices", "n_pages * page_size"
        ),
        "sinks": TensorSpec(
            description="Per-head logit added to the normalisation only.",
            shape=("n_heads",),
            dtype=_FLOAT32,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            present_when="True",
        ),
        "extra_kv_cache": TensorSpec(
            description="The extra KV pool, the same packed slots.",
            shape=("n_extra_pages", "extra_page_size", 1, 584),
            dtype=_KV_POOL,
            present_when="has_extra_kv_cache",
            distribution=Deepseekv4Fp8KvCacheDistribution(),
            lifetime="weight",
        ),
        "extra_indices": _slots(
            "Slots of the extra KV pool each token reads; negative: skipped.",
            "n_extra_indices",
            "n_extra_pages * extra_page_size",
            "has_extra_indices",
        ),
        "rope_freqs": TensorSpec(
            description="The RoPE table: a (c, s) per pair of each position.",
            shape=("n_positions", ROPE_DIM),
            dtype=_FLOAT32,
            present_when="has_rope_freqs",
            distribution=_FINITE,
            lifetime="weight",
        ),
        "positions": TensorSpec(
            description="Each token's row of the RoPE table.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            present_when="has_positions",
            distribution=IntRangeDistribution(lo="0", hi="n_positions - 1"),
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each head's attended values, the RoPE part rotated back with rope_freqs.",
            shape=("n_tokens", 1, "n_heads", 512),
            dtype=_BF16,
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(
                detect_facts=(LayoutFact(location="arg", name="kv_cache_layout", expected_value="row_major"),)
            ),
            # sglang slices each pool page out of a row padded to 576 bytes
            "padded_pages": LayoutVariant(
                layouts={"kv_cache": Deepseekv4KvPages(), "extra_kv_cache": Deepseekv4KvPages()},
                detect_facts=(
                    LayoutFact(location="arg", name="kv_cache_layout", expected_value="padded_pages"),
                ),
            ),
        }
    ),
    compare=CompareSpec(determinism="bounded"),
    reference=Deepseekv41SparseMlaReference(),
    baseline=Deepseekv41SparseMlaBaseline(),
)

DEEPSEEKV41_MXFP8_LINEAR = OpContract(
    contract_id="deepseekv41_mxfp8_linear@1.0",
    description=(
        "sglang's MXFP8 linear of a DeepSeek-V4.1 layer on gfx950 (aiter_mxfp8_blockscaled_linear, and "
        "each group of the wo_a projection's BMM), "
        "before any bias: x @ weight.T, weight fp8 e4m3 at an e8m0 scale per 32 x 32 block. A bf16 "
        "x is first rounded to fp8 e4m3 at an e8m0 scale per 32 values of a row, the scale "
        "2**ceil(log2(amax / 448)) of the block's largest magnitude, never below 1e-30; an fp8 x "
        "comes with those scales."
    ),
    op_type="deepseekv41",
    axes={
        "m": axis("Rows of x.", 1, 2**20, is_var=True),
        "n": axis("Output columns.", MXFP8_BLOCK, 262144, multiple_of=MXFP8_BLOCK),
        "k": axis("Reduction length.", MXFP8_BLOCK, 65536, multiple_of=MXFP8_BLOCK),
        "scale_rows": derived_axis("Weight-scale rows.", f"n // {MXFP8_BLOCK}"),
        "scale_columns": derived_axis("Scale columns.", f"k // {MXFP8_BLOCK}"),
    },
    attrs={"has_x_scale": _presence("x_scale", "x_scale")},
    constraints=(
        "(x.dtype == 'float8_e4m3fn') == has_x_scale",
        # sglang quantizes the wo_a projection's input before the call
        "layout_variant != 'shuffled_16x16' or has_x_scale",
    ),
    inputs={
        "x": TensorSpec(
            description="Activations: bf16, or the fp8 values x_scale scales.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT8_E4M3FN)),
            distribution=_FINITE,
        ),
        "weight": TensorSpec(
            description="Weights.",
            shape=("n", "k"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "weight_scale": TensorSpec(
            description="Per 32 x 32 block of weight.",
            shape=("scale_rows", "scale_columns"),
            dtype=_E8M0,
            # as the MXFP4 MoE ops draw theirs: 2**-5 to 2**1 times 1 / sqrt(fan_in)
            distribution=LogUniformDistribution(
                log2_lo="-5 - bit_length(k) // 2", log2_hi="1 - bit_length(k) // 2"
            ),
            lifetime="weight",
        ),
        "x_scale": TensorSpec(
            description="Per 32 values of a row of an fp8 x.",
            shape=("m", "scale_columns"),
            dtype=_E8M0,
            present_when="has_x_scale",
            distribution=LogUniformDistribution(log2_lo="-8", log2_hi="0"),
        ),
    },
    outputs={
        "output": TensorSpec(
            description="x @ weight.T, dequantized.",
            shape=("m", "n"),
            dtype=_BF16,
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(
                detect_facts=(LayoutFact(location="arg", name="weight_layout", expected_value="row_major"),)
            ),
            # a group of the wo_a projection: sglang holds its weight as aiter's shuffle_weight stores it
            "shuffled_16x16": LayoutVariant(
                layouts={"weight": Preshuffle(tile=(16, 16))},
                detect_facts=(
                    LayoutFact(location="arg", name="weight_layout", expected_value="shuffled_16x16"),
                ),
            ),
        }
    ),
    compare=CompareSpec(determinism="bounded"),
    reference=Deepseekv41Mxfp8LinearReference(),
    baseline=Deepseekv41Mxfp8LinearBaseline(),
)

CONTRACTS = (DEEPSEEKV41_MXFP8_LINEAR, DEEPSEEKV41_SPARSE_MLA)
