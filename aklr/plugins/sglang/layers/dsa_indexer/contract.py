# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's preparation in sglang's Indexer on HIP (srt/layers/attention/dsa/
dsa_indexer.py): the query heads and the key it scores with, before the scores. The key
goes into the layer's index-K cache; the scores and their top-k stay on sglang."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)

__all__ = [
    "CONTRACTS",
    "DSA_INDEXER_KEY_PREPARE",
    "DSA_INDEXER_PREPARE",
    "FP8_MAX",
    "HEAD_DIM",
    "PAGE_SIZE",
    "ROPE_DIM",
    "SCALE_FLOOR",
]

# An indexer head and key: the Hadamard transform's length and the cache's 128 bytes a key
HEAD_DIM = 128
# The RoPE part: a head's and a key's first 64 values, in interleaved pairs (2i, 2i + 1)
ROPE_DIM = 64
# Keys per index-K page: aiter's preshuffled layout of 16 x 16 tiles
PAGE_SIZE = 64
# A key's or head's fp8 scale: 2 ** ceil(log2(max(amax, SCALE_FLOOR) / FP8_MAX))
SCALE_FLOOR = 1e-4
FP8_MAX = 448.0

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()
# The var axes' bounds: pages of an index-K pool, rows of a rotary table
_MAX_PAGES = 2**24
_MAX_POSITIONS = 2**24

# The bounds, from fp32 emulations of sglang's path and of the kernels' split-K sums at GLM-5.2's
# shapes (1 to 256 tokens with 30 input sets each, 1024 and 4096 with 4), against the float64
# reference: the projections' other summation order rounds a few values to the next bf16, so a
# few fp8 values sit one fp8 step from the reference's, and a few rows (2 of 3.4e6 query heads,
# none of 1.1e5 keys) have their amax cross a power of two times 448, the fp8 scale then twice
# or half the reference's and the values half or twice. The query side takes SQNR, which admits
# such a row: one of a token's 32 heads leaves 15 dB, and the emulations stay at 43 dB or more.
# References broken on purpose (no or NeoX RoPE, cos and sin swapped, the position one off, no
# Hadamard transform or its 128^-0.5, a scale not a power of two, n_heads^-0.5, the scale or
# 128^-0.5 left out of head_weights) come to 10 dB or less on q_fp8 or 9 dB on head_weights.
_QUERY_BOUND = SqnrBound(min_sqnr_db=12.0)
_WEIGHTS_BOUND = SqnrBound(min_sqnr_db=10.0)
# The cache is bounded per byte, so that a scale's bytes count as much as a key's: one fp8 step
# (0.125 of a value) and 2 below a row's largest values (a few one-ulp bf16 differences summed
# by the Hadamard transform). The emulations meet it everywhere; a key missing, not
# preshuffled, without LayerNorm or its bias, or at twice its scale, and a scale not stored,
# fail it. A scale step of one key fails it too unless the pool's bytes outnumber that key's
# 128 values by more than 1 / max_violation_fraction, as the bench's pools of the fewest pages
# do from 3200 tokens on; a scale stored at exactly twice the one its values were divided by
# is not seen (its fp32 bytes differ in a sign bit of fp8 zero and at most one fp8 step).
_CACHE_BOUND = ToleranceBound(absolute_tolerance=2.0, relative_tolerance=0.125)
# The cache's bytes before the call: 64 to 256, far from a stored scale's bytes (0, 0, 0 and
# about 1.5 as fp8 values), so that a scale that is not stored fails the bound
_UNWRITTEN = LogUniformDistribution(log2_lo=6, log2_hi=8)
_COMPARE = CompareSpec(max_violation_fraction=3e-4)
# no_values has no element to compare
_EMPTY_BOUND = SqnrBound(min_sqnr_db=0.0)

_KEY_AXES = {
    "n_tokens": axis("Tokens in this call, padding rows included.", 1, 65536, is_var=True),
    "model_dim": axis("The hidden states' width: wk's and weights_proj's input.", 128, 16384),
    "n_positions": axis("Rows of the rotary tables.", 1, _MAX_POSITIONS, is_var=True, any_size=True),
    "n_pages": axis("Pages of the index-K cache.", 1, _MAX_PAGES, is_var=True, any_size=True),
}
# distinct slots: a token writes its own
_KEY_CONSTRAINTS = (f"n_tokens <= {PAGE_SIZE} * n_pages",)

_KEY_INPUTS = {
    "x": TensorSpec(
        description="Each token's hidden states, the layer's input.",
        shape=("n_tokens", "model_dim"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "wk": TensorSpec(
        description="wk: the key's projection.",
        shape=(HEAD_DIM, "model_dim"),
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "k_norm_weight": TensorSpec(
        description="k_norm's (LayerNorm) weight.",
        shape=(HEAD_DIM,),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "k_norm_bias": TensorSpec(
        description="k_norm's bias.",
        shape=(HEAD_DIM,),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "cos_cache": TensorSpec(
        description="The rotary cosines, a row per position and a column per RoPE pair.",
        shape=("n_positions", ROPE_DIM // 2),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "sin_cache": TensorSpec(
        description="The rotary sines, as cos_cache.",
        shape=("n_positions", ROPE_DIM // 2),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "positions": TensorSpec(
        description="Each token's position: its row of the rotary tables.",
        shape=("n_tokens",),
        dtype=_INT64,
        distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
    ),
    "out_cache_loc": TensorSpec(
        description=(
            f"Each token's slot of the index-K cache (page slot // {PAGE_SIZE}, key "
            f"slot % {PAGE_SIZE}), distinct but for sglang's graph padding rows, which all write "
            "slot 0, a slot no sequence reads."
        ),
        shape=("n_tokens",),
        dtype=_INT64,
        distribution=IntRangeDistribution(lo=0, hi=f"{PAGE_SIZE} * n_pages - 1", distinct=True),
    ),
    "index_k_cache": TensorSpec(
        description=(
            f"This layer's index-K cache, as sglang views it: pages of {PAGE_SIZE} keys, their "
            f"{HEAD_DIM} fp8 values in 16 x 16 tiles (16 keys by 16 values) and then their fp32 "
            "scales, each page's bytes compared as fp8 values. The tokens' keys are stored "
            "into it."
        ),
        shape=("n_pages", PAGE_SIZE, HEAD_DIM + 4),
        dtype=_FP8,
        distribution=_UNWRITTEN,
        mutable=True,
        error_bounds=_CACHE_BOUND,
    ),
    "k_norm_eps": ScalarSpec(
        description="k_norm's epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=1e-6,
    ),
}

_KEY_TEXT = (
    "Each token's key is LayerNorm(x wk^T) (k_norm_weight, k_norm_bias, k_norm_eps), its "
    f"first {ROPE_DIM} values turned by RoPE in interleaved pairs (2i, 2i + 1) by cos_cache "
    "and sin_cache at the token's position, then the Hadamard transform over its "
    f"{HEAD_DIM} values (Sylvester order) times {HEAD_DIM}^-0.5, each step rounded to bf16 as "
    "sglang rounds it; it is stored into index_k_cache at its out_cache_loc slot as fp8 "
    f"values at a scale of 2^ceil(log2(max(amax, {SCALE_FLOOR}) / {FP8_MAX:g})) of its values, "
    "and that scale in fp32."
)

DSA_INDEXER_PREPARE = OpContract(
    contract_id="dsa_indexer_prepare@1.0",
    description=(
        "sglang's DSA indexer before its scores on one rank (Indexer.forward_cuda on HIP, the "
        "Hadamard-rotated index-K format). Each query head is a head of q_lora wq_b^T, rounded "
        "to bf16, turned and transformed as the key, and quantized to q_fp8 at its own scale as "
        "the key is; its head weight is bf16(bf16(x weights_proj^T) * n_heads^-0.5) times that "
        f"scale times {HEAD_DIM}^-0.5. {_KEY_TEXT}"
    ),
    op_type="dsa_indexer",
    axes={
        **_KEY_AXES,
        "q_lora_rank": axis("q_lora's width: wq_b's input.", 128, 16384),
        "n_heads": axis("Indexer heads.", 1, 128),
        "index_cols": derived_axis("wq_b's rows: the heads side by side.", f"n_heads * {HEAD_DIM}"),
    },
    constraints=_KEY_CONSTRAINTS,
    inputs={
        "x": _KEY_INPUTS["x"],
        "q_lora": TensorSpec(
            description="Each token's query latent after q_a_layernorm: wq_b's input.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "wq_b": TensorSpec(
            description="wq_b: the query heads' projection, each head's rows in turn.",
            shape=("index_cols", "q_lora_rank"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="q_lora_rank"),
            lifetime="weight",
        ),
        "wk": _KEY_INPUTS["wk"],
        "weights_proj": TensorSpec(
            description="weights_proj: each head's weight, a row per head.",
            shape=("n_heads", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        **{n: s for n, s in _KEY_INPUTS.items() if n not in ("x", "wk")},
    },
    outputs={
        "q_fp8": TensorSpec(
            description="Each token's query heads, turned and transformed, in fp8 at their scales.",
            shape=("n_tokens", "n_heads", HEAD_DIM),
            dtype=_FP8,
            error_bounds=_QUERY_BOUND,
        ),
        "head_weights": TensorSpec(
            description="Each token's head weights, each with its head's fp8 scale folded in.",
            shape=("n_tokens", "n_heads"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=_WEIGHTS_BOUND,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

DSA_INDEXER_KEY_PREPARE = OpContract(
    contract_id="dsa_indexer_key_prepare@1.0",
    description=(
        "sglang's DSA indexer's key-only path on one rank (Indexer._forward_cuda_k_only on HIP, "
        "the Hadamard-rotated index-K format), which stores the tokens' keys and computes no "
        f"query. {_KEY_TEXT}"
    ),
    op_type="dsa_indexer",
    axes=_KEY_AXES,
    constraints=_KEY_CONSTRAINTS,
    inputs=_KEY_INPUTS,
    outputs={
        "no_values": TensorSpec(
            description=(
                "No values: the op's one result is index_k_cache, and a contract declares an output."
            ),
            shape=(0,),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=_EMPTY_BOUND,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

CONTRACTS = (DSA_INDEXER_PREPARE, DSA_INDEXER_KEY_PREPARE)
