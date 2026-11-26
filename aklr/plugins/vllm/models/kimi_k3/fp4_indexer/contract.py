# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's DSA indexer prefill with an MXFP4 index cache, as vLLM runs it on ROCm under DCP
(``models/kimi_k3/amd/ops/fp4_indexer_prefill.py``, ``prefill_indexer``): one query tile of this
rank's rows of a prefill chunk scored against the chunk's keys, gathered from every DCP rank, and
each row's top-k keys. vLLM quantizes the queries (its Triton kernel), scores them (aiter's FlyDSL
``pa_mqa_logits_fp4_prefill``), selects (aiter's ``top_k_per_row_prefill``) and makes the ids
relative to each row's first key; this op is those four steps on one rank."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
)
from aklr.plugins.vllm.models.kimi_k3.fp4_indexer.baseline import KimiK3Fp4IndexerPrefillBaseline
from aklr.plugins.vllm.models.kimi_k3.fp4_indexer.reference import (
    HEAD_DIM,
    KEYS_PER_PAGE,
    MXFP4_GROUP,
    SCALE_FLOOR,
    KimiK3Fp4IndexerPrefillReference,
)

__all__ = [
    "CONTRACTS",
    "HEAD_DIM",
    "KEYS_PER_PAGE",
    "KIMI_K3_FP4_INDEXER_PREFILL",
    "MXFP4_GROUP",
    "SCALE_FLOOR",
]

# vLLM sizes each query tile to keep its float32 scores under 1 GiB: rows * 64 * n_pages * 4 bytes
_SCORE_BYTES = 2**30
# Kimi-K3.1's weight scale, n_heads^-0.5 * HEAD_DIM^-0.5 at 32 heads: 2^-6 in float32
_WEIGHT_SCALE = 0.015625

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()

_QUANTIZATION = (
    "Each query head is quantized as vLLM's _quantize does, per group of 32 values: the scale is "
    f"2^ceil(log2(max(amax / 6, {SCALE_FLOOR:g}))) (amax / 6 in float32, the exponent at most 127), "
    "each value divided by it and rounded to the nearest E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), a "
    "tie to the even code."
)
_KEY_LAYOUT = (
    f"Key t of the chunk is slot s = t % {KEYS_PER_PAGE} of page t // {KEYS_PER_PAGE}, aiter's row-group "
    "layout: its values 32g..32g + 31 are key_values[page, g, s] (two E2M1 values a byte, the lower "
    "nibble first), their E8M0 scale key_scales[page, g, (s % 16) * 4 + s // 16]."
)

KIMI_K3_FP4_INDEXER_PREFILL = OpContract(
    contract_id="kimi_k3_fp4_indexer_prefill@1.0",
    description=(
        "vLLM's Kimi-K3.1 DSA indexer prefill on one rank with an MXFP4 index cache (prefill_indexer, "
        "one query tile): each row's score of key t is weight_scale * sum over heads h of "
        "relu(q_h . k_t) * weights[h], q_h the row's MXFP4 head, k_t the MXFP4 key, both decoded exactly, "
        "the dot product and the sum in float32; the row's keys are t in [row_starts, row_ends); "
        "indices holds the row's topk highest-scoring keys, as t - row_starts, in any order (vLLM's "
        "top-k writes them unsorted); a row of at most topk keys holds every key, in any order, and -1 "
        "in its other slots. "
        f"{_QUANTIZATION} {_KEY_LAYOUT}"
    ),
    op_type="fp4_indexer",
    axes={
        # vLLM's tiles: this rank's rows of a chunk, at most 8192 (2048 at K3.1's 16384 tokens a step, DCP8)
        "n_tokens": axis("Query rows in this call.", 1, 8192, is_var=True),
        "n_heads": axis("Index heads: 16, 32 or 64 (aiter's MFMA tiles).", 16, 64, multiple_of=16),
        "n_pages": axis(
            f"Pages of {KEYS_PER_PAGE} staged keys: every key the chunk's requests hold, from every "
            "DCP rank, in order.",
            1,
            2**20,
            is_var=True,
            any_size=True,
        ),
        "topk": axis("Keys selected per row.", 1, 2048),
    },
    constraints=(
        "n_heads != 48",
        f"n_tokens * n_pages * {KEYS_PER_PAGE * 4} < {_SCORE_BYTES}",
    ),
    inputs={
        "q": TensorSpec(
            description="Each row's index query heads, wq_index's output.",
            shape=("n_tokens", "n_heads", HEAD_DIM),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "weights": TensorSpec(
            description="Each row's head weights, wv_index's output.",
            shape=("n_tokens", "n_heads"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "key_values": TensorSpec(
            description=(
                "The staged keys' E2M1 values, 16 bytes of a key's group a row (aiter's pages, "
                "[n_pages, 1, 4, 64, 16] bytes in vLLM)."
            ),
            shape=("n_pages", HEAD_DIM // MXFP4_GROUP, KEYS_PER_PAGE, MXFP4_GROUP),
            dtype=DTypeSpec(any_of=(DType.FLOAT4_E2M1,)),
            distribution=_FINITE,
        ),
        "key_scales": TensorSpec(
            description=(
                "The staged keys' E8M0 group scales, a page's 64 per group in aiter's transposed "
                "16 x 4 order ([n_pages, 1, 4, 64] in vLLM)."
            ),
            shape=("n_pages", HEAD_DIM // MXFP4_GROUP, KEYS_PER_PAGE),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E8M0,)),
            # a LayerNorm'd key's groups: amax / 6 within a few octaves of 1
            distribution=LogUniformDistribution(log2_lo=-3, log2_hi=1),
        ),
        "row_starts": TensorSpec(
            description="Each row's first key: its request's first key in the chunk.",
            shape=("n_tokens",),
            dtype=_INT32,
            # requests' offsets: a few keys in, below every row's end
            distribution=IntRangeDistribution(lo=0, hi="n_pages - 1"),
        ),
        "row_ends": TensorSpec(
            description="Each row's end, past its last key (causal: its position + 1).",
            shape=("n_tokens",),
            dtype=_INT32,
            # this rank's rows of a chunk of DCP8 ranks' rows: within the last 8 * n_tokens keys
            distribution=IntRangeDistribution(
                lo=f"max(n_pages, {KEYS_PER_PAGE} * n_pages - 8 * n_tokens)", hi=f"{KEYS_PER_PAGE} * n_pages"
            ),
        ),
        "weight_scale": ScalarSpec(
            description="The head and softmax scales folded into every weight.",
            dtype=DType.FLOAT32,
            interval=(2.0**-60, 2.0**60),
            default=_WEIGHT_SCALE,
        ),
    },
    outputs={
        "indices": TensorSpec(
            description="Each row's selected keys, relative to its first, -1 past a short row's keys.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
            # exact once the reference's check has put both in one order and taken a tied key for the
            # reference's
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # Scores summed in another order than the reference's differ in their last bits, so a key whose
    # score ties the row's topk-th may stand for it: the reference's check compares each row's keys as
    # a set, by their scores.
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=KimiK3Fp4IndexerPrefillReference(),
    baseline=KimiK3Fp4IndexerPrefillBaseline(),
)

CONTRACTS = (KIMI_K3_FP4_INDEXER_PREFILL,)
