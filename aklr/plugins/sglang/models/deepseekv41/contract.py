# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's calls on one rank of its TP group: sglang's own where it serves them without
a monokernel (every prefill, and the decode and verify steps a monokernel does not take), and
the launches of its AMD monokernel, on the steps it takes too.

A model is a scenario: DeepSeek-V4.1-Flash at TP2 is 32 query heads on a rank."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    Deepseekv4Fp8KvCacheDistribution,
    Deepseekv4KvPages,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    GateUpInterleavedScales,
    GateUpInterleavedWeight,
    IntRangeDistribution,
    LecunNormalDistribution,
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

__all__ = [
    "CONTRACTS",
    "DEEPSEEKV41_MONO_K1",
    "DEEPSEEKV41_MONO_K2_ATTENTION",
    "DEEPSEEKV41_MONO_K2_FFN",
    "DEEPSEEKV41_MXFP8_LINEAR",
    "DEEPSEEKV41_SPARSE_MLA",
    "GROUP_HEADS",
    "HEAD_DIM",
    "KV_SCALE_BLOCK",
    "MXFP8_BLOCK",
    "ROPE_DIM",
]

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
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
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
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

# A head's values: the position-free part, then the RoPE part
HEAD_DIM = 512
# Position-free values of a KV record sharing one scale
KV_SCALE_BLOCK = 64
_NOPE_DIM = HEAD_DIM - ROPE_DIM
# Heads sharing one group of the output projection
GROUP_HEADS = 8
_UINT8 = DTypeSpec(any_of=(DType.UINT8,))
# about [0.5, 1.5], as rms_norm's weights
_NORM_WEIGHT = LogUniformDistribution(log2_lo="-1", log2_hi="0.58")
_MIX = LogUniformDistribution(log2_lo="-3", log2_hi="0")
# bf16 values of fp32 sums: a few bf16 roundings of the value, and an absolute part where a sum cancels
_ROUNDED = ToleranceBound(absolute_tolerance=2**-7, relative_tolerance=2**-6)
# the mixes are fp32 throughout; the streams they read differ by bf16 rounding flips only
_MIXES = ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-3)
# the mixes of streams holding the attention's output, itself bounded by energy: sglang's chain
# is 55 to 68 dB from the reference on gfx950
_ATTENDED_MIXES = SqnrBound(min_sqnr_db=40.0)
# outputs behind an MXFP8 rounding: a flipped fp8 code moves a whole element, so by energy
_QUANTIZED = SqnrBound(min_sqnr_db=30.0)
# fp8 codes of bf16 math: one fp8 ulp where a value crosses an fp8 rounding boundary
_FP8_CODES = ToleranceBound(absolute_tolerance=2**-5, relative_tolerance=2**-3)


def _epsilon(description: str, default: float) -> ScalarSpec:
    return ScalarSpec(
        description=description, dtype=DType.FLOAT32, interval=_POSITIVE_FLOAT32, default=default
    )


def _weight_scale(rows: str, columns: str, description: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=(rows, columns),
        dtype=_E8M0,
        distribution=LogUniformDistribution(log2_lo="-5 - bit_length(hidden_size) // 2", log2_hi="-6"),
        lifetime="weight",
    )


def _seam_outputs(
    streams: ToleranceBound | SqnrBound, mixes: ToleranceBound | SqnrBound
) -> dict[str, TensorSpec]:
    def mix(description: str, *shape: int) -> TensorSpec:
        return TensorSpec(
            description=description, shape=("n_tokens", *shape), dtype=_FLOAT32, error_bounds=mixes
        )

    return {
        "residual_out": TensorSpec(
            description="The streams after the owed post; the streams as given when none is owed.",
            shape=("n_tokens", 4, "hidden_size"),
            dtype=_BF16,
            error_bounds=streams,
        ),
        "pre_mix_out": mix("This boundary's pre mix, which the next boundary sums the streams under.", 4),
        "post_mix_out": mix("This boundary's post mix, of the post it owes.", 4),
        "comb_mix_out": mix("This boundary's combination mix, of the post it owes.", 4, 4),
    }


_SEAM_AXES = {
    "hidden_size": axis("The hidden size.", MXFP8_BLOCK, 16384, multiple_of=MXFP8_BLOCK),
    "mix_dim": derived_axis("The four streams flattened: the mix projection's reduction.", "4 * hidden_size"),
    "hidden_blocks": derived_axis("MXFP8 blocks of a hidden row.", f"hidden_size // {MXFP8_BLOCK}"),
}
_STREAMS = TensorSpec(
    description="The four residual streams.",
    shape=("n_tokens", 4, "hidden_size"),
    dtype=_BF16,
    distribution=_FINITE,
)
_SEAM_WEIGHTS = {
    "hc_fn": TensorSpec(
        description="Projects the flattened streams to the 24 mix logits: 4 pre, 4 post, 16 combination.",
        shape=(24, "mix_dim"),
        dtype=_FLOAT32,
        distribution=LecunNormalDistribution(fan_in="mix_dim"),
        lifetime="weight",
    ),
    "hc_scale": TensorSpec(
        description="Scale of the pre, post and combination logits.",
        shape=(3,),
        dtype=_FLOAT32,
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
        lifetime="weight",
    ),
    "hc_base": TensorSpec(
        description="Bias of the 24 mix logits.",
        shape=(24,),
        dtype=_FLOAT32,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "norm_weight": TensorSpec(
        description="Weight of the RMSNorm of the mixed streams.",
        shape=("hidden_size",),
        dtype=_BF16,
        distribution=_NORM_WEIGHT,
        lifetime="weight",
    ),
    "hc_rms_eps": _epsilon("Added to the mean square of the streams.", 1e-6),
    "hc_eps": _epsilon("Added to each pre mix and in each Sinkhorn normalization.", 1e-6),
    "norm_eps": _epsilon("Added to each RMSNorm's mean square.", 1e-6),
    "sinkhorn_iters": ScalarSpec(
        description="Sinkhorn normalizations of the combination mix.",
        dtype=DType.INT32,
        interval=(1, 2**31 - 1),
        default=20,
    ),
}


_SEAM = (
    "residual_out[j] = x * post_mix[j] + sum_i comb_mix[i, j] * residual[i], rounded to bf16; the "
    "layer input is sum_j mix[j] * residual_out[j] under the previous boundary's pre mix, rounded "
    "to bf16, then times rsqrt(mean square + norm_eps) and norm_weight. This boundary's mixes "
    "come of the 24 logits hc_fn @ flatten(residual_out) * rsqrt(mean square + hc_rms_eps): "
    "sigmoid(logits[:4] * hc_scale[0] + hc_base[:4]) + hc_eps the pre mix, twice the sigmoid of "
    "the next 4 at hc_scale[1] the post mix, and the last 16 at hc_scale[2] as a 4 x 4 softmax "
    "over its rows plus hc_eps, normalized over columns, then sinkhorn_iters - 1 times over rows "
    "and columns (each sum plus hc_eps), the combination mix."
)

DEEPSEEKV41_MONO_K1 = OpContract(
    contract_id="deepseekv41_mono_k1@1.0",
    description=(
        "The attention front of a DeepSeek-V4.1 layer on one rank, as one launch (the K1 of "
        "sglang's AMD monokernel, on this rank's query heads): the mHC boundary into the "
        "attention, the input RMSNorm, wqkv_a, the query through q_norm, wq_b and RoPE, and the "
        "token's KV record. " + _SEAM + " Without x no post is owed; without prev_pre_mix (the "
        "first layer) the layer input is stream 0. The normed input, rounded to bf16 then to fp8 "
        "e4m3 at an e8m0 scale per 32 values (2**ceil(log2(amax / 448))), times wqkv is the qkv "
        "latent, rounded to bf16: q_rank columns of q's latent, then 512 of the KV latent. q's "
        "latent under its RMSNorm is q_latent; quantized the same way and times wq_b it is q, "
        "rounded to bf16, the last 64 values of every head then rotated in pairs (a, b) by the "
        "pair's (c, s) at the token's position: (a c - b s, a s + b c). The KV latent under its "
        "RMSNorm, its last 64 values rotated alike, is the record: k_rope those 64, and k_nope "
        "the first 448 as fp8 e4m3 at a scale per 64 values, 2**ceil(log2(max(amax, 1e-4) / 448)), "
        "whose e8m0 bytes are k_nope_scale."
    ),
    op_type="deepseekv41",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        **_SEAM_AXES,
        "q_rank": axis("q's latent.", MXFP8_BLOCK, 8192, multiple_of=MXFP8_BLOCK),
        "q_dim": axis(
            "Every head on this rank's query, flattened.", HEAD_DIM, 128 * HEAD_DIM, multiple_of=HEAD_DIM
        ),
        "n_positions": axis("Rows of the RoPE table.", 1, 2**21, is_var=True, any_size=True),
        "n_heads": derived_axis("Query heads on this rank.", f"q_dim // {HEAD_DIM}"),
        "qkv_dim": derived_axis("A qkv latent row: q's latent, then the KV latent.", f"q_rank + {HEAD_DIM}"),
        "qkv_blocks": derived_axis("MXFP8 block rows of wqkv.", f"(q_rank + {HEAD_DIM}) // {MXFP8_BLOCK}"),
        "q_blocks": derived_axis("MXFP8 block rows of wq_b.", f"q_dim // {MXFP8_BLOCK}"),
        "q_rank_blocks": derived_axis("MXFP8 blocks of a q latent row.", f"q_rank // {MXFP8_BLOCK}"),
    },
    attrs={
        "has_x": _presence("x", "x"),
        "has_post_mix": _presence("post_mix", "post"),
        "has_comb_mix": _presence("comb_mix", "comb"),
        "has_prev_pre_mix": _presence("prev_pre_mix", "pre"),
    },
    constraints=("has_x == has_post_mix", "has_x == has_comb_mix"),
    inputs={
        "x": TensorSpec(
            description="The previous layer's FFN output, whose post is owed.",
            shape=("n_tokens", "hidden_size"),
            dtype=_BF16,
            present_when="has_x",
            distribution=_FINITE,
        ),
        "residual": _STREAMS,
        "post_mix": TensorSpec(
            description="The owed post's mix of x into each stream.",
            shape=("n_tokens", 4),
            dtype=_FLOAT32,
            present_when="has_post_mix",
            distribution=_MIX,
        ),
        "comb_mix": TensorSpec(
            description="The owed post's mix of the streams: [token, from, to].",
            shape=("n_tokens", 4, 4),
            dtype=_FLOAT32,
            present_when="has_comb_mix",
            distribution=_MIX,
        ),
        "prev_pre_mix": TensorSpec(
            description="The previous boundary's pre mix.",
            shape=("n_tokens", 4),
            dtype=_FLOAT32,
            present_when="has_prev_pre_mix",
            distribution=_MIX,
        ),
        **_SEAM_WEIGHTS,
        "wqkv": TensorSpec(
            description="wqkv_a's weight.",
            shape=("qkv_dim", "hidden_size"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "wqkv_scale": _weight_scale("qkv_blocks", "hidden_blocks", "Per 32 x 32 block of wqkv."),
        "q_norm_weight": TensorSpec(
            description="q_norm's weight.",
            shape=("q_rank",),
            dtype=_BF16,
            distribution=_NORM_WEIGHT,
            lifetime="weight",
        ),
        "kv_norm_weight": TensorSpec(
            description="kv_norm's weight.",
            shape=(HEAD_DIM,),
            dtype=_BF16,
            distribution=_NORM_WEIGHT,
            lifetime="weight",
        ),
        "wq_b": TensorSpec(
            description="wq_b's weight: a head's 512 rows after another's.",
            shape=("q_dim", "q_rank"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "wq_b_scale": _weight_scale("q_blocks", "q_rank_blocks", "Per 32 x 32 block of wq_b."),
        "rope_freqs": TensorSpec(
            description="The RoPE table: a (c, s) per pair of each position.",
            shape=("n_positions", ROPE_DIM),
            dtype=_FLOAT32,
            distribution=_MIX,
            lifetime="weight",
        ),
        "positions": TensorSpec(
            description="Each token's row of the RoPE table.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            distribution=IntRangeDistribution(lo="0", hi="n_positions - 1"),
        ),
    },
    outputs={
        "q": TensorSpec(
            description="Each token's query per head, the RoPE part rotated.",
            shape=("n_tokens", "n_heads", HEAD_DIM),
            dtype=_BF16,
            error_bounds=_QUANTIZED,
        ),
        **_seam_outputs(_ROUNDED, _MIXES),
        "normed": TensorSpec(
            description="The layer input under its RMSNorm: the compressor's and indexer's input.",
            shape=("n_tokens", "hidden_size"),
            dtype=_BF16,
            error_bounds=_ROUNDED,
        ),
        "q_latent": TensorSpec(
            description="q's latent under q_norm: the indexer's input.",
            shape=("n_tokens", "q_rank"),
            dtype=_BF16,
            error_bounds=_QUANTIZED,
        ),
        "k_nope": TensorSpec(
            description="The record's position-free part, in units of its scales.",
            shape=("n_tokens", _NOPE_DIM),
            dtype=_FP8,
            error_bounds=_FP8_CODES,
        ),
        "k_nope_scale": TensorSpec(
            description="E8M0 exponent bytes at [token, block of 64]; a scale is 2**(byte - 127).",
            shape=("n_tokens", _NOPE_DIM // KV_SCALE_BLOCK),
            dtype=_UINT8,
        ),
        "k_rope": TensorSpec(
            description="The record's RoPE part, rotated.",
            shape=("n_tokens", ROPE_DIM),
            dtype=_BF16,
            error_bounds=_QUANTIZED,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

DEEPSEEKV41_MONO_K2_ATTENTION = OpContract(
    contract_id="deepseekv41_mono_k2_attention@1.0",
    description=(
        "The attention back of a DeepSeek-V4.1 layer and its FFN boundary on one rank, as one "
        "launch (the K2 of sglang's AMD monokernel up to the MoE's input; under data-parallel "
        "attention the MoE's rows are gathered from the other ranks first). "
        "deepseekv41_sparse_mla's attention with the rotation back, rounded to bf16; each group "
        "of 8 heads times its rows of wo_a, rounded to bf16; that, to fp8 e4m3 at an e8m0 scale "
        "per 32 values, times wo_b, rounded to bf16: x, the attention's output on a rank holding "
        "every head. A rank of a TP group holds its share of the heads, of wo_a's groups and of "
        "wo_b's columns: x is then its wo_b output plus peer_x, the sum of the other ranks', "
        "which the caller's collective brings, rounded to bf16. Then the mHC boundary into the "
        "FFN and its RMSNorm, ffn_input: " + _SEAM
    ),
    op_type="deepseekv41",
    axes={
        **DEEPSEEKV41_SPARSE_MLA.axes,
        **_SEAM_AXES,
        "o_dim": axis(
            "wo_a's rows, a group's after another's: wo_b's reduction.",
            MXFP8_BLOCK,
            65536,
            multiple_of=MXFP8_BLOCK,
        ),
        "o_blocks": derived_axis("MXFP8 blocks of wo_b's reduction.", f"o_dim // {MXFP8_BLOCK}"),
    },
    attrs={**DEEPSEEKV41_SPARSE_MLA.attrs, "has_peer_x": _presence("peer_x", "peer")},
    constraints=(
        *DEEPSEEKV41_SPARSE_MLA.constraints,
        "has_rope_freqs",
        f"n_heads % {GROUP_HEADS} == 0",
        f"o_dim % (n_heads // {GROUP_HEADS}) == 0",
    ),
    inputs={
        **DEEPSEEKV41_SPARSE_MLA.inputs,
        "wo_a": TensorSpec(
            description="wo_a's weight over a group's 8 heads.",
            shape=("o_dim", GROUP_HEADS * HEAD_DIM),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in=GROUP_HEADS * HEAD_DIM),
            lifetime="weight",
        ),
        "wo_b": TensorSpec(
            description="wo_b's weight.",
            shape=("hidden_size", "o_dim"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "wo_b_scale": _weight_scale("hidden_blocks", "o_blocks", "Per 32 x 32 block of wo_b."),
        "peer_x": TensorSpec(
            description="The other ranks' wo_b outputs of these tokens, each in bf16, summed.",
            shape=("n_tokens", "hidden_size"),
            dtype=_FLOAT32,
            present_when="has_peer_x",
            distribution=_FINITE,
        ),
        "residual": _STREAMS,
        "post_mix": TensorSpec(
            description="The attention boundary's post mix, of x into each stream.",
            shape=("n_tokens", 4),
            dtype=_FLOAT32,
            distribution=_MIX,
        ),
        "comb_mix": TensorSpec(
            description="The attention boundary's combination mix: [token, from, to].",
            shape=("n_tokens", 4, 4),
            dtype=_FLOAT32,
            distribution=_MIX,
        ),
        "pre_mix": TensorSpec(
            description="The attention boundary's pre mix, which the FFN's input sums the streams under.",
            shape=("n_tokens", 4),
            dtype=_FLOAT32,
            distribution=_MIX,
        ),
        **_SEAM_WEIGHTS,
    },
    outputs={
        "ffn_input": TensorSpec(
            description="The MoE's input: the mixed streams under the post-attention RMSNorm.",
            shape=("n_tokens", "hidden_size"),
            dtype=_BF16,
            error_bounds=_QUANTIZED,
        ),
        **_seam_outputs(_QUANTIZED, _ATTENDED_MIXES),
    },
    layout_variants=DEEPSEEKV41_SPARSE_MLA.layout_variants,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))


def _ffn_layout_facts(shuffled: bool, scale_rank: int) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": scale_rank}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


DEEPSEEKV41_MONO_K2_FFN = OpContract(
    contract_id="deepseekv41_mono_k2_ffn@1.0",
    description=(
        "The MoE of a DeepSeek-V4.1 layer on one rank, as sglang's DeepseekV2MoE.forward_normal "
        "runs it where no monokernel takes the step, and the monokernel's K2 after the FFN seam "
        "but for the shared expert, which K2 reads from the checkpoint's FP8 weights: on "
        "hidden_states (the rows of every rank under DP attention), the router's "
        "logits in fp32; each token's topk experts by sqrt(softplus(logit)) plus correction_bias, "
        "weighed by the unbiased scores over their sum times routed_scaling_factor, and the "
        "shared expert, the last slot, at weight 1; each expert down(silu(min(gate(x), limit)) "
        "* clamp(up(x), -limit, limit)) of its MXFP4 weights, both linears' inputs in MXFP8 "
        "(e4m3 at an e8m0 scale per 32), as aiter's a8w4 experts take them; output is their sum "
        "on this rank, before sglang sums the ranks'."
    ),
    op_type="deepseekv41",
    axes={
        "n_tokens": axis("Rows in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    # the target's 6 of 384, and the DSpark draft's 3 of 128
    attrs={"topk": attr("Routed experts per token.", 6, 3, param_name="topk")},
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": TensorSpec(
            description="The MoE's input: the FFN seam's normed rows.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "router_weight": TensorSpec(
            description="Router projection.",
            shape=("n_routed", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "correction_bias": TensorSpec(
            description="Added to the scores to pick experts; never weighs them.",
            shape=("n_routed",),
            # bf16, as sglang holds it for aiter's sqrtsoftplus gate
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=LogUniformDistribution(
                log2_lo="-5 - bit_length(model_dim) // 2", log2_hi="1 - bit_length(model_dim) // 2"
            ),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=LogUniformDistribution(
                log2_lo="-5 - bit_length(inter_dim) // 2", log2_hi="1 - bit_length(inter_dim) // 2"
            ),
            lifetime="weight",
        ),
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies the routed weights only: DeepSeek-V4.1's.",
            dtype=DType.FLOAT32,
            values=(1.5,),
            default=1.5,
        ),
        "swiglu_limit": ScalarSpec(
            description="The gate is at most this, the up projection within +/- this.",
            dtype=DType.FLOAT32,
            values=(10.0,),
            default=10.0,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="The experts' output on this rank, before sglang sums the ranks'.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            # sglang at 35 dB up to the 64 rows its split-K router serves; past them its bf16
            # router GEMM re-routes about 5 rows in a hundred of the bench's inputs: 22 dB
            error_bounds=SqnrBound(min_sqnr_db=18.0),
        ),
    },
    # row-major, or as sglang holds the experts for aiter's a8w4 path: gate and up rows
    # interleaved and shuffled, the scales flat
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(layouts={}, detect_facts=_ffn_layout_facts(False, 3)),
            "gate_up_interleaved": LayoutVariant(
                layouts={
                    "w1": GateUpInterleavedWeight(gate_up=True),
                    "w2": GateUpInterleavedWeight(gate_up=False),
                    "w1_scale": GateUpInterleavedScales(gate_up=True),
                    "w2_scale": GateUpInterleavedScales(gate_up=False),
                },
                detect_facts=_ffn_layout_facts(True, 2),
            ),
        }
    ),
    # a near-tie in the top-k routes a token to another expert
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (
    DEEPSEEKV41_MONO_K1,
    DEEPSEEKV41_MONO_K2_ATTENTION,
    DEEPSEEKV41_MONO_K2_FFN,
    DEEPSEEKV41_MXFP8_LINEAR,
    DEEPSEEKV41_SPARSE_MLA,
)
