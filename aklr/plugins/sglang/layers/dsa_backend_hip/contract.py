# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's DSA sparse MLA on ROCm over its fp8_ds_mla KV pool, as sglang 0.5.24.dev809 runs it on gfx950:
the prefill's ``sparse_mla_prefill_gfx950``, which ``triton_sparse_mla_fwd`` calls for a prefill or extend
wider than the GPU's CUs. vLLM's ``sparse_mla_fp8ds`` attends the same 656-byte rows.

A model is a scenario: Kimi-K3.1-Pro at TP8 has 12 heads a rank and a top-k of 2048."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    Fp8DsMlaRowsDistribution,
    IntRangeDistribution,
    SqnrBound,
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
from aklr.plugins.sglang.layers.dsa_backend_hip.baseline import DsaSparseMlaFp8DsPrefillBaseline
from aklr.plugins.sglang.layers.dsa_backend_hip.reference import DsaSparseMlaFp8DsPrefillReference

__all__ = ["CONTRACTS", "DSA_SPARSE_MLA_FP8_DS_PREFILL", "KV_ROW_BYTES"]

# an fp8_ds_mla row: 512 fp8 NoPE codes, an fp32 scale per 128 of them, 64 bf16 RoPE values
KV_ROW_BYTES = 656
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FINITE = FiniteDistribution()
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# sglang's gfx950 kernel, the baseline, addresses the pool by 31-bit byte offsets
# (supports_sparse_mla_prefill_gfx950); a larger pool stays on triton, which the route leaves to sglang
_POOL_ROWS = (2**31 - 1) // KV_ROW_BYTES

# Not dsa_sparse_mla_prefill@1.0 (dsa_backend): that op reads 576-byte fp8 rows, unscaled, against one fp8 q,
# without the LSE.
DSA_SPARSE_MLA_FP8_DS_PREFILL = OpContract(
    contract_id="dsa_sparse_mla_fp8_ds_prefill@1.0",
    description=(
        "sglang's ROCm sparse MLA prefill on Kimi-K3.1's fp8_ds_mla pool (sparse_mla_prefill_gfx950, which "
        "triton_sparse_mla_fwd runs for a prefill or extend wider than the GPU's CUs): each head of each token "
        "attends the pool rows its top-k slots name (-1: none), a key being the row's 512 NoPE values, fp8 times "
        "their 128-group's fp32 scale, then its 64 bf16 RoPE values, with scores (q_nope, q_rope) . key * "
        "sm_scale; output is the softmax-weighted sum of the keys' NoPE values and lse the natural log of the sum "
        "of exp(score); a token that reads no key gets 0 and -inf."
    ),
    op_type="dsa_backend_hip",
    axes={
        "n_tokens": axis("Query tokens of the prefill or extend.", 1, 65536, is_var=True),
        # the heads supports_sparse_mla_prefill_gfx950 takes
        "n_heads": axis("Query heads on this rank.", 1, 16),
        "topk": axis("Entries per token of indices: the indexer's top-k.", 1, 65536),
        "n_slots": axis("Rows of this rank's KV pool.", 1, _POOL_ROWS, is_var=True, any_size=True),
    },
    constraints=("topk <= n_slots",),
    inputs={
        "q_nope": TensorSpec(
            description="Each token's query per head in the latent's basis: q_nope times W_UK.",
            shape=("n_tokens", "n_heads", 512),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "q_rope": TensorSpec(
            description="Each token's RoPE'd query part per head.",
            shape=("n_tokens", "n_heads", 64),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "kv_cache": TensorSpec(
            description=(
                "This rank's KV pool, a row a slot in the fp8_ds_mla layout: 512 fp8 e4m3 NoPE values, an "
                "fp32 scale per 128 of them, a power of two as Kimi-K3.1's ue8m0 cache writer keeps it, then "
                "64 bf16 RoPE values."
            ),
            shape=("n_slots", KV_ROW_BYTES),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            distribution=Fp8DsMlaRowsDistribution(),
        ),
        "indices": TensorSpec(
            description="Each token's top-k slots in the pool; -1 names no key (a token with fewer candidates).",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: Kimi-K3.1's 192**-0.5.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=192**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each head's softmax-weighted sum of its keys' NoPE values.",
            shape=("n_tokens", "n_heads", 512),
            dtype=_BF16,
            # a kernel may round q's NoPE part to fp8 per (head, 128), as aiter's FlyDSL sparse MLA does
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
        "lse": TensorSpec(
            description="Natural log of each head's sum of exp(score).",
            shape=("n_tokens", "n_heads"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=0.25, relative_tolerance=0.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=DsaSparseMlaFp8DsPrefillReference(),
    baseline=DsaSparseMlaFp8DsPrefillBaseline(),
)

CONTRACTS = (DSA_SPARSE_MLA_FP8_DS_PREFILL,)
