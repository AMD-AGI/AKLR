# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BaselineRelativeBound,
    Deepseekv4Fp8KvCacheDistribution,
    Deepseekv4KvPages,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
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
    derived_axis,
)

__all__ = ["CONTRACTS", "DEEPSEEK_V4_ATTENTION_Q_PREP", "DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE"]


_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

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
    op_type="deepseek_v4",
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

CONTRACTS = (DEEPSEEK_V4_ATTENTION_Q_PREP, DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE)
