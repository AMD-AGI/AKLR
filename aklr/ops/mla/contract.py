# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    Deepseekv4Fp8KvCacheDistribution,
    Deepseekv4KvPages,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
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
    ShapeByAttrSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["CONTRACTS", "FLASH_MLA_WITH_KVCACHE", "FP8_PAGED_MQA_LOGITS"]

_FINITE_DISTRIBUTION = FiniteDistribution()
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
_KV_CACHE_FORMATS = ("dsv4_fp8", "fp8_e4m3", "bf16")

# Each sparse pool is pages of page_size slots: 576-byte payloads, then 8-byte scale slots.
_PAGE_SIZE_CONSTRAINT = "{0} == 2 or {0} == 64 or {0} == 128 or {0} == 256"
_SLOT_LIMIT = 2**31  # slots are int32 indices
# The DSA models' context, 2^20 tokens, and room for the draft tokens sglang's tables add past it
_MQA_CONTEXT_LIMIT = 2**20 + 4096
_MQA_PAGE_SIZES = (
    16,
    32,
    64,
    128,
    256,
)  # aiter's preshuffled pages hold 16-token tiles


def _by_format(sparse: tuple[str | int, ...], dense: tuple[str | int, ...]) -> ShapeByAttrSpec:
    return ShapeByAttrSpec(
        attr_name="kv_cache_format",
        cases={"dsv4_fp8": sparse, "fp8_e4m3": dense, "bf16": dense},
    )


def _sparse_indices(description: str, n_indices: str, slots: str, present_when: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("batch", 1, n_indices),
        dtype=_INT32,
        present_when=present_when,
        distribution=IntRangeDistribution(lo="-1", hi=f"{slots} - 1"),
    )


def _sparse_lens(description: str, n_indices: str, present_when: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("batch",),
        dtype=_INT32,
        present_when=present_when,
        distribution=IntRangeDistribution(lo="0", hi=n_indices),
    )


def _kv_cache_layout_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="kv_cache_layout", expected_value=value),)


def _presence(tensor: str, abbrev: str | None = None) -> AttrSpec:
    return attr(f"Whether {tensor} is passed.", False, True, default=False, abbrev=abbrev)


FLASH_MLA_WITH_KVCACHE = OpContract(
    contract_id="flash_mla_with_kvcache@1.0",
    description=(
        "MLA attention over a paged KV pool, FlashMLA's API. Sparse: each token "
        "attends to the slots its indices name, in the pool and the optional extra "
        "one; sinks join the normalisation only. Dense: each request attends to its "
        "context rows; q's rope part is last, scores are q . row * sm_scale * "
        "kv_scale and values row[:head_dim_v] * kv_scale. With causal, query i of a "
        "request of length L reads its first L - query_length + i + 1 rows."
    ),
    op_type="mla",
    axes={
        "batch": axis(
            "Rows: one token each (sparse) or one request each (dense).",
            1,
            65536,
            is_var=True,
        ),
        "query_length": axis("Queries per row.", 1, 65536),
        "n_heads": axis("Query heads.", 1, 128),
        "n_pages": axis("Pages of the KV pool.", 1, _SLOT_LIMIT, is_var=True),
        "page_size": axis("Slots per page of the KV pool.", 1, 256),
        "n_indices": axis("Sparse indices per token.", 64, 65536, multiple_of=64),
        "n_extra_pages": axis("Pages of the extra KV pool.", 1, _SLOT_LIMIT, is_var=True),
        "extra_page_size": axis("Slots per page of the extra KV pool.", 2, 256),
        "n_extra_indices": axis("Extra sparse indices per token.", 64, 65536, multiple_of=64),
        "table_width": axis(
            "Columns of the 2-D block_table, or length of the flat one.",
            1,
            _SLOT_LIMIT - 1,  # cu_seqlens_k holds it as an int32 offset
            is_var=True,
        ),
        "n_offsets": axis("Offsets in cu_seqlens_k: batch + 1.", 2, 65537, is_var=True),
    },
    attrs={
        "causal": attr(
            "Whether query i of a request of length L reads only its first L - query_length + i + 1 rows.",
            False,
            True,
            default=False,
            param_name="causal",
        ),
        "kv_cache_format": attr(
            "The pool rows: DeepSeek-V4's packed fp8 slots (sparse), or 576 fp8 or bf16 values (dense).",
            *_KV_CACHE_FORMATS,
            default="dsv4_fp8",
            param_name="kv_cache_format",
        ),
        "return_lse": attr(
            "Whether lse is returned too.",
            False,
            True,
            default=False,
            param_name="return_lse",
            abbrev="lse",
        ),
        # short names keep the longest schema name a file name (<= 255 bytes)
        "has_sparse_indices": _presence("sparse_indices", "sparse"),
        "has_sparse_lens": _presence("sparse_lens", "lens"),
        "has_extra_kv_cache": _presence("extra_kv_cache", "extra"),
        "has_extra_sparse_indices": _presence("extra_sparse_indices", "extra_idx"),
        "has_extra_sparse_lens": _presence("extra_sparse_lens", "extra_lens"),
        "has_block_table": _presence("block_table"),
        "has_cache_seqlens": _presence("cache_seqlens"),
        "has_cu_seqlens_k": _presence("cu_seqlens_k"),
        "has_sinks": _presence("sinks", "sinks"),
    },
    constraints=(
        "return_lse",  # an output is never optional: output alone is not modelled
        # sparse indices read the dsv4 pool, a block table a dense one
        "has_sparse_indices == (kv_cache_format == 'dsv4_fp8')",
        "has_sparse_indices != has_block_table",
        "has_block_table == (has_cache_seqlens or has_cu_seqlens_k)",
        "not (has_cache_seqlens and has_cu_seqlens_k)",
        "has_sparse_indices or (not (has_sparse_lens or has_extra_kv_cache or has_sinks))",
        # the extra pool and its indices come together; its lengths only with them
        "has_extra_kv_cache == has_extra_sparse_indices",
        "has_extra_kv_cache or (not has_extra_sparse_lens)",
        "(kv_cache.dtype == 'bfloat16') == (kv_cache_format == 'bf16')",
        "kv_cache_format != 'fp8_e4m3' or kv_cache.dtype == 'float8_e4m3fn'",
        "has_sparse_indices or page_size == 1",  # a dense pool row is a page
        f"(not has_sparse_indices) or ({_PAGE_SIZE_CONSTRAINT.format('page_size')})",
        _PAGE_SIZE_CONSTRAINT.format("extra_page_size"),
        f"n_pages * page_size <= {_SLOT_LIMIT}",
        f"n_extra_pages * extra_page_size <= {_SLOT_LIMIT}",
        # aiter's fp8 kernel takes 16-head blocks; 64 and 128 heads take a bf16-only path
        "has_sparse_indices or (n_heads <= 96 and n_heads != 64)",
        "(not has_sparse_indices) or n_heads == 64 or n_heads == 128",
        # only a causal flat table takes several queries; fp8 decodes, bf16 verifies
        "query_length == 1 or (causal and has_cu_seqlens_k)",
        "not (causal and has_sparse_indices)",
        "(kv_cache_format == 'bf16') == (query_length > 1)",
        "(not has_cu_seqlens_k) or n_offsets == batch + 1",
        "(not has_cu_seqlens_k) or table_width >= batch * query_length",
        # sglang's dsv4 pool is padded; a dense pool row is a whole page
        "(kv_cache_format == 'dsv4_fp8') == (layout_variant == 'padded_pages')",
    ),
    inputs={
        "q": TensorSpec(
            description="The queries of each row; dense: the rope part last.",
            shape=_by_format(
                ("batch", "query_length", "n_heads", 512),
                ("batch", "query_length", "n_heads", 576),
            ),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "kv_cache": TensorSpec(
            description="The KV pool: packed dsv4 slots, or one 576-value row a page.",
            shape=_by_format(("n_pages", "page_size", 1, 584), ("n_pages", "page_size", 1, 576)),
            dtype=DTypeSpec(any_of=(DType.UINT8, DType.FLOAT8_E4M3FN, DType.BFLOAT16)),
            # the import check takes a predicate reading an attr both ways, so the dtype-only items come first
            distribution=(
                ("kv_cache.dtype == 'uint8'", Deepseekv4Fp8KvCacheDistribution()),
                ("kv_cache.dtype == 'bfloat16'", FiniteDistribution()),
                ("kv_cache_format == 'dsv4_fp8'", Deepseekv4Fp8KvCacheDistribution()),
                (None, FiniteDistribution()),
            ),
            lifetime="weight",
        ),
        "sparse_indices": _sparse_indices(
            "Slots of the KV pool each token reads; negative: skipped.",
            "n_indices",
            "n_pages * page_size",
            "has_sparse_indices",
        ),
        "sparse_lens": _sparse_lens(
            "Leading sparse_indices each token reads; all when absent.",
            "n_indices",
            "has_sparse_lens",
        ),
        "extra_kv_cache": TensorSpec(
            description="The extra KV pool, packed dsv4 slots.",
            shape=("n_extra_pages", "extra_page_size", 1, 584),
            dtype=DTypeSpec(any_of=(DType.UINT8, DType.FLOAT8_E4M3FN)),
            present_when="has_extra_kv_cache",
            distribution=Deepseekv4Fp8KvCacheDistribution(),
            lifetime="weight",
        ),
        "extra_sparse_indices": _sparse_indices(
            "Slots of the extra KV pool each token reads; negative: skipped.",
            "n_extra_indices",
            "n_extra_pages * extra_page_size",
            "has_extra_sparse_indices",
        ),
        "extra_sparse_lens": _sparse_lens(
            "Leading extra_sparse_indices each token reads; all when absent.",
            "n_extra_indices",
            "has_extra_sparse_lens",
        ),
        "block_table": TensorSpec(
            description=(
                "Pool rows of each request: a (batch, width) table read up to "
                "cache_seqlens, or every request's rows back to back."
            ),
            shape=ShapeByAttrSpec(
                attr_name="has_cache_seqlens",
                cases={"true": ("batch", "table_width"), "false": ("table_width",)},
            ),
            dtype=_INT32,
            present_when="has_block_table",
            distribution=IntRangeDistribution(lo="0", hi="n_pages * page_size - 1"),
        ),
        "cache_seqlens": TensorSpec(
            description="Leading block_table entries each request reads.",
            shape=("batch",),
            dtype=_INT32,
            present_when="has_cache_seqlens",
            # aiter assumes every request reads at least one row
            distribution=IntRangeDistribution(lo="1", hi="table_width"),
        ),
        "cu_seqlens_k": TensorSpec(
            description="Where each request's rows start in block_table, then its length.",
            shape=("n_offsets",),
            dtype=_INT32,
            present_when="has_cu_seqlens_k",
            distribution=CuSeqlensDistribution(min_seqlen="query_length", total="table_width"),
        ),
        "head_dim_v": ScalarSpec(
            description="Values per output head: 512 dense, None sparse.",
            dtype=DType.INT32,
            values=(512,),
            present_when="not has_sparse_indices",
        ),
        # always passed: present_when gives the api default None, which declines
        "sm_scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            present_when="True",
        ),
        "sinks": TensorSpec(
            description="Per-head logit added to the normalisation only, not to lse.",
            shape=("n_heads",),
            dtype=_FLOAT32,
            present_when="has_sinks",
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "kv_scale": ScalarSpec(
            description="Multiplies every KV value, in the logits and the values.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=1.0,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Softmax-weighted sum of the read values; 0 for no row read.",
            shape=("batch", "query_length", "n_heads", 512),
            dtype=DTypeSpec(same_as_tensor="q"),
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
        "lse": TensorSpec(
            description=(
                "Log-sum-exp of the scaled logits, sinks excluded; for no row read "
                "+inf (sparse) or -inf (dense)."
            ),
            shape=("batch", "query_length", "n_heads"),
            dtype=_FLOAT32,
            error_bounds=ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-3),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_kv_cache_layout_facts("row_major")),
            # sglang slices each pool page out of a row padded to 576 bytes
            "padded_pages": LayoutVariant(
                layouts={
                    "kv_cache": Deepseekv4KvPages(),
                    "extra_kv_cache": Deepseekv4KvPages(),
                },
                detect_facts=_kv_cache_layout_facts("padded_pages"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

FP8_PAGED_MQA_LOGITS = OpContract(
    contract_id="fp8_paged_mqa_logits@1.0",
    description=(
        "DeepSeek-V4 indexer logits: for each request and each cached token t, "
        "sum over heads h of weights[h] * relu(q[h] . k_t) * scale_t, over a paged "
        "pool of fp8 keys with an fp32 scale per token."
    ),
    op_type="mla",
    axes={
        "batch": axis("Requests, one decode query each.", 1, 65536, is_var=True),
        "n_heads": axis("Indexer heads.", 1, 128),
        "n_pages": axis("Pages of the key pool.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
        "max_blocks": axis(
            "Pages per request in the block table.", 1, _MQA_CONTEXT_LIMIT, is_var=True, any_size=True
        ),
        "max_seq_len": axis(
            "Logits per request: every token of its block table row.",
            1,
            _MQA_CONTEXT_LIMIT,
            is_var=True,
            any_size=True,
        ),
    },
    attrs={
        "page_size": attr("Tokens per page.", *_MQA_PAGE_SIZES, param_name="page_size"),
        "preshuffled": attr(
            "Whether each page's keys are in aiter's 16 x 16 preshuffled tiles.",
            True,
            default=True,
            param_name="preshuffled",
        ),
    },
    # aiter writes whole pages: a narrower row would spill into the next one
    constraints=("max_seq_len == max_blocks * page_size",),
    inputs={
        "q": TensorSpec(
            description="The decode query of each request.",
            shape=("batch", 1, "n_heads", 128),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "kv_cache": TensorSpec(
            description=(
                "The key pool, a page a row: page_size x 128 fp8 keys in 16 x 16 "
                "tiles, then page_size fp32 scales."
            ),
            shape=ShapeByAttrSpec(
                attr_name="page_size",
                cases={str(p): ("n_pages", p, 1, 132) for p in _MQA_PAGE_SIZES},
            ),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            # shortcut: bytes below 0x40 keep keys and scales finite and at most 2;
            # a packed-pool distribution if a kernel needs realistic magnitudes
            distribution=IntRangeDistribution(lo="0", hi="63"),
            lifetime="weight",
        ),
        "weights": TensorSpec(
            description="Per-head weight of each request.",
            shape=("batch", "n_heads"),
            dtype=_FLOAT32,
            distribution=_FINITE_DISTRIBUTION,
        ),
        "context_lens": TensorSpec(
            description="Cached tokens of each request.",
            shape=("batch",),
            dtype=_INT32,
            # shortcut: full block tables only; past a context aiter leaves the logits unspecified
            distribution=IntRangeDistribution(lo="max_seq_len", hi="max_seq_len"),
        ),
        "block_tables": TensorSpec(
            description="The pages of each request, in context order.",
            shape=("batch", "max_blocks"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_pages - 1"),
        ),
    },
    outputs={
        "logits": TensorSpec(
            description="The logits of each request's cached tokens.",
            shape=("batch", "max_seq_len"),
            dtype=_FLOAT32,
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (FLASH_MLA_WITH_KVCACHE, FP8_PAGED_MQA_LOGITS)
