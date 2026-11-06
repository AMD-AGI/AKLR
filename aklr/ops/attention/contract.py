# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BoundedCuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    KvSplitPages,
    StridedRows,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["CONTRACTS", "PAGED_ATTENTION_DECODE", "UNIFIED_ATTENTION"]

_FINITE_DISTRIBUTION = FiniteDistribution()
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
_SLOT_LIMIT = 2**31  # slots are int32 indices


def _kv_cache_storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="kv_cache_storage", expected_value=value),)


def _kv_cache(side: str, query: str = "query") -> TensorSpec:
    return TensorSpec(
        description=f"The {side} of every cached token, block by block.",
        shape=("n_blocks", "block_size", "n_kv_heads", "head_dim"),
        dtype=DTypeSpec(same_as_tensor=query),
        distribution=_FINITE_DISTRIBUTION,
    )


# the kernel reads the cache and the table through block indices: any size serves
_N_BLOCKS = axis("Blocks of the KV cache.", 1, _SLOT_LIMIT, is_var=True, any_size=True)
_MAX_BLOCKS = axis(
    "Blocks per request in the block table.",
    1,
    _SLOT_LIMIT,
    is_var=True,
    any_size=True,
)


PAGED_ATTENTION_DECODE = OpContract(
    contract_id="paged_attention_decode@1.0",
    description=(
        "Decode attention over a paged KV cache: query head h of each request "
        "attends to KV head h // (n_heads / n_kv_heads) of the first context_lens "
        "tokens its block_tables row names, block_size tokens a block; with "
        "sliding_window > 0 only to the last sliding_window of them. Logits are "
        "q . k * scale."
    ),
    op_type="attention",
    axes={
        "batch": axis("Requests, one query token each.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads.", 1, 128),
        "n_kv_heads": axis("KV heads.", 1, 128),
        "head_dim": axis("Values per head.", 64, 256),
        "n_blocks": _N_BLOCKS,
        "block_size": axis("Tokens per block.", 16, 32),
        "max_blocks": _MAX_BLOCKS,
    },
    attrs={
        "kv_cache_dtype": attr("The cache's dtype: query's (auto).", "auto", param_name="kv_cache_dtype"),
        "kv_cache_layout": attr(
            "The cache's dim order: blocks, tokens, heads, values (NHD).",
            "NHD",
            param_name="kv_cache_layout",
        ),
        "has_alibi_slopes": attr("Whether alibi_slopes is passed.", False, True, default=False),
    },
    constraints=(
        "n_heads % n_kv_heads == 0",
        "n_heads // n_kv_heads <= 16",
        "head_dim == 64 or head_dim == 128 or head_dim == 256",
        "block_size == 16 or block_size == 32",
        f"n_blocks * block_size <= {_SLOT_LIMIT}",
        "not has_alibi_slopes",  # aiter's alibi bias is not modelled
    ),
    inputs={
        "query": TensorSpec(
            description="The query of each request.",
            shape=("batch", "n_heads", "head_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "key_cache": _kv_cache("keys"),
        "value_cache": _kv_cache("values"),
        "block_tables": TensorSpec(
            description="The blocks of each request, in context order.",
            shape=("batch", "max_blocks"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_blocks - 1"),
        ),
        "context_lens": TensorSpec(
            description="Cached tokens of each request, the query's own included.",
            shape=("batch",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="1", hi="max_blocks * block_size"),
        ),
        "scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
        ),
        "alibi_slopes": TensorSpec(
            description="Per-head alibi slope.",
            shape=("n_heads",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            present_when="has_alibi_slopes",
            distribution=_FINITE_DISTRIBUTION,
        ),
        "sliding_window": ScalarSpec(
            description="Trailing tokens each query reads; 0 reads the whole context.",
            dtype=DType.INT32,
            interval=(0, _SLOT_LIMIT - 1),
            default=0,
        ),
        "logits_soft_cap": ScalarSpec(
            description="Logit soft cap; 0 is none.",
            dtype=DType.FLOAT32,
            values=(0.0,),
            default=0.0,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="Softmax-weighted sum of the read values.",
            shape=("batch", "n_heads", "head_dim"),
            dtype=DTypeSpec(same_as_tensor="query"),
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_kv_cache_storage_facts("row_major")),
            # vLLM's: K and V halves of one cache, the query sliced from a wider buffer
            "kv_split_pages": LayoutVariant(
                layouts={
                    "query": StridedRows(),
                    "key_cache": KvSplitPages(),
                    "value_cache": KvSplitPages(),
                },
                detect_facts=_kv_cache_storage_facts("kv_split_pages"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

UNIFIED_ATTENTION = OpContract(
    contract_id="unified_attention@1.0",
    description=(
        "Causal varlen attention of a batch of requests' new tokens over a paged KV "
        "cache, vLLM's multi-token decode: request i owns query rows cu_seqlens_q[i] "
        "to cu_seqlens_q[i + 1] and the first seqused_k[i] tokens its block_table row "
        "names, block_size tokens a block, its new tokens last. Query j of a request "
        "of L new tokens sits at position seqused_k - L + j and reads the tokens at or "
        "before it; with window_size[0] >= 0 only those at most window_size[0] before "
        "it (causal: window_size[1] is not read). Query head h reads KV head h // "
        "(n_heads / n_kv_heads); logits are q . k * softmax_scale."
    ),
    op_type="attention",
    axes={
        "total_q": axis("Query rows of every request.", 1, 2**20, is_var=True),
        "batch": axis("Requests.", 1, 65536, is_var=True),
        "n_offsets": axis("Offsets in cu_seqlens_q: batch + 1.", 2, 65537, is_var=True),
        "n_heads": axis("Query heads.", 1, 128),
        "n_kv_heads": axis("KV heads.", 1, 128),
        "head_dim": axis("Values per head.", 64, 256),
        "n_blocks": _N_BLOCKS,
        "block_size": axis("Tokens per block.", 16, 64),
        "max_blocks": _MAX_BLOCKS,
    },
    attrs={
        # vLLM's decode_max_query_len; aiter reads 1 as one query per request
        "max_seqlen_q": attr(
            "The longest request's new tokens.",
            *range(1, 9),
            param_name="max_seqlen_q",
            abbrev="max_q",
        ),
        # vLLM's (sliding_window - 1, 0), or (-1, -1) without one
        "window_size": attr(
            "Earlier tokens a query reads beyond itself, then later ones; -1 is all.",
            (-1, -1),
            (2047, 0),
            param_name="window_size",
            abbrev="window",
        ),
        "causal": attr("Causal masking: aiter's only mode.", True, param_name="causal"),
        **{
            f"has_{side}_descale": attr(f"Whether {side}_descale is passed.", False, True, default=False)
            for side in ("q", "k", "v")
        },
    },
    constraints=(
        "n_offsets == batch + 1",
        "batch <= total_q",
        "total_q <= batch * max_seqlen_q",
        "n_heads % n_kv_heads == 0",
        "n_heads // n_kv_heads <= 16",
        "head_dim == 64 or head_dim == 128 or head_dim == 256",
        "block_size == 16 or block_size == 32 or block_size == 64",
        f"n_blocks * block_size <= {_SLOT_LIMIT}",
        "max_seqlen_q <= max_blocks * block_size",
        # aiter scales the logits and values by the descales' first element
        "not has_q_descale",
        "not has_k_descale",
        "not has_v_descale",
    ),
    inputs={
        "q": TensorSpec(
            description="The new tokens' queries, request after request.",
            shape=("total_q", "n_heads", "head_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "k": _kv_cache("keys", "q"),
        "v": _kv_cache("values", "q"),
        "cu_seqlens_q": TensorSpec(
            description="Where each request's query rows start, then total_q.",
            shape=("n_offsets",),
            dtype=_INT32,
            distribution=BoundedCuSeqlensDistribution(total="total_q", max_seqlen="max_seqlen_q"),
        ),
        "seqused_k": TensorSpec(
            description="Cached tokens of each request, its new tokens included.",
            shape=("batch",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="max_seqlen_q", hi="max_blocks * block_size"),
        ),
        "block_table": TensorSpec(
            description="The blocks of each request, in context order.",
            shape=("batch", "max_blocks"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_blocks - 1"),
        ),
        "softmax_scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
        ),
        "softcap": ScalarSpec(
            description="Logit soft cap; 0 is none.",
            dtype=DType.FLOAT32,
            values=(0.0,),
            default=0.0,
        ),
        **{
            f"{side}_descale": TensorSpec(
                description=f"Dequantization scale of {what}.",
                shape=shape,
                dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
                present_when=f"has_{side}_descale",
                distribution=_FINITE_DISTRIBUTION,
            )
            for side, what, shape in (
                ("q", "q", (1,)),
                ("k", "k, per request and KV head", ("batch", "n_kv_heads")),
                ("v", "v, per request and KV head", ("batch", "n_kv_heads")),
            )
        },
    },
    outputs={
        "out": TensorSpec(
            description="Softmax-weighted sum of the read values, per query row.",
            shape=("total_q", "n_heads", "head_dim"),
            dtype=DTypeSpec(same_as_tensor="q"),
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_kv_cache_storage_facts("row_major")),
            # vLLM's: K and V halves of one cache, the query sliced from a wider buffer
            "kv_split_pages": LayoutVariant(
                layouts={"q": StridedRows(), "k": KvSplitPages(), "v": KvSplitPages()},
                detect_facts=_kv_cache_storage_facts("kv_split_pages"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (PAGED_ATTENTION_DECODE, UNIFIED_ATTENTION)
