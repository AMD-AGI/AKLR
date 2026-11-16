# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    ElementOrder,
    FiniteDistribution,
    IntRangeDistribution,
    PermutationDistribution,
    SqnrBound,
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

__all__ = ["CONTRACTS", "DSA_SPARSE_MLA_DECODE", "DSA_TOPK_TRANSFORM", "ROPE_DIM"]

_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# The KV pool's bytes that 32-bit buffer offsets address: rows past them stay on sglang.
_POOL_BYTES = 2**31 - 2
# The DSA models' context, 2^20 tokens, and room for the draft tokens sglang's tables add past it
_CONTEXT_LIMIT = 2**20 + 4096
# The RoPE part of a query head and of a KV row: the DSA models' (sglang's kernel takes a
# power of 2)
ROPE_DIM = 64
# GLM-5's softmax scale, 1 / sqrt(nope_dim + rope_dim): the default bench's inputs take
_SM_SCALE = 0.0625

DSA_SPARSE_MLA_DECODE = OpContract(
    contract_id="dsa_sparse_mla_decode@1.0",
    description=(
        "sglang's DSA backend's sparse MLA decode on one rank, kv_b_proj absorbed "
        "(DeepseekSparseAttnBackend._forward_triton_decode): each head of each token attends "
        "the KV pool rows its top-k slots name (-1: none) with scores q . row * sm_scale, q "
        "and the rows the fp8 values sglang holds, unscaled; the output is the "
        "softmax-weighted sum of the rows' latent parts, 0 for a token that reads no row."
    ),
    op_type="dsa_backend",
    axes={
        "n_tokens": axis("Tokens in this call, one query each.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "kv_dim": axis("A query head and a KV row: the latent, then 64 RoPE values.", 192, 576),
        "topk": axis("Top-k entries per token.", 1, 65536),
        "n_slots": axis("Rows of the KV pool.", 1, 2**31 - 1, is_var=True, any_size=True),
        "kv_lora_rank": derived_axis(
            "The latent's width: a row's first part, and the output's.",
            f"kv_dim - {ROPE_DIM}",
        ),
    },
    constraints=(
        "kv_lora_rank % 128 == 0",
        # distinct top-k entries, -1 among them
        "topk <= n_slots + 1",
        f"n_slots * kv_dim <= {_POOL_BYTES}",
    ),
    inputs={
        "q": TensorSpec(
            description=(
                "Each token's query per head: q_nope times W_UK (the latent's basis), then the RoPE'd q_pe."
            ),
            shape=("n_tokens", "n_heads", "kv_dim"),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "kv_cache": TensorSpec(
            description=("This layer's KV pool: a row per slot, the latent then the RoPE part."),
            shape=("n_slots", "kv_dim"),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "topk_indices": TensorSpec(
            description="The slots each token attends; -1: none.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=_SM_SCALE,
        ),
    },
    outputs={
        "output": TensorSpec(
            description=(
                "Each head's output in the latent's basis: the rows' latent parts "
                "weighed by the softmax of their scores."
            ),
            shape=("n_tokens", "n_heads", "kv_lora_rank"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            # sglang's own kernel, which rounds the softmax weights to fp8, comes to about
            # 33 dB on these inputs
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

DSA_TOPK_TRANSFORM = OpContract(
    contract_id="dsa_topk_transform@1.0",
    description=(
        "sglang's DSA indexer top-k on HIP (sgl_kernel.fast_topk_transform_fused): each row "
        "selects the topk highest of its first lengths scores and writes their slots in the "
        "page_size=1 page table of its sequence, the cu_seqlens_q range holding the row; a row "
        "of at most topk scores writes all of its slots in order, then -1."
    ),
    op_type="dsa_backend",
    axes={
        "n_tokens": axis("Rows, one per token.", 1, 65536, is_var=True),
        "n_candidates": axis("Scores per row.", 1, _CONTEXT_LIMIT, is_var=True, any_size=True),
        "n_seqs": axis("Sequences.", 1, 65536, is_var=True, any_size=True),
        "n_starts": axis("Sequence boundaries: n_seqs + 1.", 2, 65537, is_var=True, any_size=True),
        "n_slots": axis("Page table columns.", 1, _CONTEXT_LIMIT, is_var=True, any_size=True),
        "topk": axis("Slots per row, 2048: all sglang's kernel takes.", 2048, 2048),
    },
    constraints=("n_starts == n_seqs + 1", "n_seqs <= n_tokens"),
    inputs={
        "score": TensorSpec(
            description="Candidate scores per row.",
            shape=("n_tokens", "n_candidates"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=PermutationDistribution(),
        ),
        "lengths": TensorSpec(
            description="Valid candidates per row, a prefix its page table covers.",
            shape=("n_tokens",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="min(n_candidates, n_slots)"),
        ),
        "page_table_size_1": TensorSpec(
            description="Each sequence's slot of each candidate position, distinct within a row.",
            shape=("n_seqs", "n_slots"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_slots - 1", distinct=True),
        ),
        "cu_seqlens_q": TensorSpec(
            description="Each sequence's first row, then n_tokens.",
            shape=("n_starts",),
            dtype=_INT32,
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_tokens"),
        ),
    },
    outputs={
        "out_page_indices": TensorSpec(
            description="Selected slots; -1 after a short row's entries.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # Ties and a full row's order are the kernel's choice.
    compare=CompareSpec(
        determinism=DeterminismLevel.BOUNDED,
        element_order=ElementOrder.ANY_IN_FULL_ROWS,
    ),
)

CONTRACTS = (DSA_SPARSE_MLA_DECODE, DSA_TOPK_TRANSFORM)
