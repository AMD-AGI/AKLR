# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    Deepseekv4Fp8KvCacheDistribution,
    Deepseekv4KvPages,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["CONTRACTS", "DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

# A KV pool row (584 bytes per token): 448 fp8 NoPE values, 64 bf16 RoPE values, then, after
# the page's rows, a UE8M0 exponent per 64 NoPE values and a pad byte.
_HEAD_DIM, _ROPE_DIM, _TOKEN_BYTES = 512, 64, 584

_PAGE_SIZE_CONSTRAINT = "{0} == 2 or {0} == 64 or {0} == 128 or {0} == 256"

_SLOT_LIMIT = 2**31  # slots are int32


def _kv_pool(description: str, n_pages: str, page_size: str, present_when: str | None = None) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=(n_pages, page_size, 1, _TOKEN_BYTES),
        dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN, DType.UINT8)),
        present_when=present_when,
        distribution=Deepseekv4Fp8KvCacheDistribution(),
        lifetime="weight",
    )


def _kv_cache_layout_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="kv_cache_layout", expected_value=value),)


def _slots(
    description: str, width: str, n_pages: str, page_size: str, present_when: str | None = None
) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("n_tokens", width),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
        present_when=present_when,
        distribution=IntRangeDistribution(lo=-1, hi=f"{n_pages} * {page_size} - 1"),
    )


DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE = OpContract(
    contract_id="deepseek_v4_sparse_mla_decode_inv_rope@1.0",
    description=(
        "sglang's aiter_sparse_decode_fwd on DeepSeek-V4's fp8 KV pools with inv_rope: each "
        "head of each query row attends the slots its sparse_indices name in the SWA pool and, "
        "when given, its extra_sparse_indices in the extra (compressed) pool (negative: none), "
        "scores q . row * sm_scale, the head's sink joining the normalisation only; the output "
        "is the softmax-weighted sum of the whole 512-wide rows (0 for a row that reads no "
        "slot), its last 64 lanes then rotated back by the row's position: per (cos, sin) "
        "pair of rope_freqs[positions], even * cos + odd * sin and odd * cos - even * sin."
    ),
    op_type="hip_flash_mla",
    axes={
        "n_tokens": axis(
            "Query rows in this call: requests times tokens per request.", 1, 65536, is_var=True
        ),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "n_pages": axis("Pages of the SWA pool.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
        "page_size": axis("Slots per page of the SWA pool.", 2, 256),
        "n_indices": axis("SWA slots per query row.", 64, 65536, multiple_of=64),
        "n_extra_pages": axis("Pages of the extra pool.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
        "extra_page_size": axis("Slots per page of the extra pool.", 2, 256),
        "n_extra_indices": axis("Extra-pool slots per query row.", 64, 65536, multiple_of=64),
        "n_positions": axis("Rows of the RoPE table.", 1, _SLOT_LIMIT, is_var=True, any_size=True),
    },
    attrs={
        "has_extra_kv_cache": attr(
            "Whether the extra pool and its slots are passed (compressed layers).",
            False,
            True,
            default=False,
            abbrev="extra",
        ),
        "has_extra_sparse_indices": attr(
            "Whether the extra pool's slots are passed.", False, True, default=False, abbrev="extra_idx"
        ),
    },
    constraints=(
        # the extra pool and its slots come together
        "has_extra_kv_cache == has_extra_sparse_indices",
        _PAGE_SIZE_CONSTRAINT.format("page_size"),
        _PAGE_SIZE_CONSTRAINT.format("extra_page_size"),
        f"n_pages * page_size <= {_SLOT_LIMIT}",
        f"n_extra_pages * extra_page_size <= {_SLOT_LIMIT}",
    ),
    inputs={
        "q": TensorSpec(
            description="Each query row's heads: q_nope in the latent's basis, then the RoPE'd q_pe.",
            shape=("n_tokens", "n_heads", _HEAD_DIM),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
        "kv_cache": _kv_pool("This layer's SWA pool, packed fp8 slots.", "n_pages", "page_size"),
        "sparse_indices": _slots(
            "The SWA slots each query row reads; negative: skipped.", "n_indices", "n_pages", "page_size"
        ),
        "extra_kv_cache": _kv_pool(
            "This layer's extra (compressed) pool, packed fp8 slots.",
            "n_extra_pages",
            "extra_page_size",
            "has_extra_kv_cache",
        ),
        "extra_sparse_indices": _slots(
            "The extra-pool slots each query row reads; negative: skipped.",
            "n_extra_indices",
            "n_extra_pages",
            "extra_page_size",
            "has_extra_sparse_indices",
        ),
        "sinks": TensorSpec(
            description="Per-head logit added to the normalisation only.",
            shape=("n_heads",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, _FLOAT32_LARGEST),
            default=_HEAD_DIM**-0.5,
        ),
        "rope_freqs": TensorSpec(
            description="Per position, the 32 (cos, sin) pairs of the RoPE, interleaved.",
            shape=("n_positions", _ROPE_DIM),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "positions": TensorSpec(
            description="Each query row's position: its row of rope_freqs.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
    },
    outputs={
        "output": TensorSpec(
            description=(
                "Each head's softmax-weighted sum of the rows it reads, the last 64 lanes "
                "inverse-rotated by the row's position."
            ),
            shape=("n_tokens", "n_heads", _HEAD_DIM),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            # flash_mla_with_kvcache's bound; aiter's kernel (bf16 weights and dot operands)
            # peaks at about a third of it on these inputs
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_kv_cache_layout_facts("row_major")),
            # sglang's pools slice each page out of a row padded to 576 bytes
            "padded_pages": LayoutVariant(
                layouts={"kv_cache": Deepseekv4KvPages(), "extra_kv_cache": Deepseekv4KvPages()},
                detect_facts=_kv_cache_layout_facts("padded_pages"),
            ),
        }
    ),
)

CONTRACTS = (DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE,)
