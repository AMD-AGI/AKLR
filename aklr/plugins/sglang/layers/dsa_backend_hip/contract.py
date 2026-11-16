# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's ROCm DSA partials over one DCP rank's fp8_ds_mla KV shard (srt/layers/attention/dsa/
dsa_backend_hip.py), Kimi-K3.1-Pro's: a scenario is the heads the DCP group gathers on a rank, 96 at
TP8 / DCP8, and the global top-k's width, 2048."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    Fp8DsMlaRowsDistribution,
    IntRangeDistribution,
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
)

__all__ = ["CONTRACTS", "DSA_SPARSE_MLA_DCP_PARTIAL"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# sglang runs its gfx950 kernel, this op's baseline, only on a pool its 32-bit byte offsets address
# (supports_sparse_mla_decode_gfx950); on a larger one forward_triton_dcp_partial runs triton, a branch
# the route leaves to sglang. The bound is the replaced branch's, which every kernel of the op serves in
# full, not one kernel's limit.
_POOL_ROWS = (2**31 - 1) // 656

# Not dsa_sparse_mla_decode@1.0: that op reads 576-byte fp8 rows against one fp8 q, with neither the LSE nor a
# length per row the DCP merge needs.
DSA_SPARSE_MLA_DCP_PARTIAL = OpContract(
    contract_id="dsa_sparse_mla_dcp_partial@1.0",
    description=(
        "sglang's ROCm DSA decode or target-verify partial on one DCP rank, past its top-k "
        "localization (dsa_backend_hip.forward_triton_dcp_partial on gfx950): each head of each row "
        "attends the pool rows its first topk_length slots name, a key being the row's 512 NoPE "
        "values, fp8 times their 128-group's fp32 scale, then its 64 bf16 RoPE values, with scores "
        "(q_nope, q_rope) . key * sm_scale; output is the softmax-weighted sum of the keys' NoPE "
        "values and lse the natural log of the sum of exp(score), which the DCP merge weighs the "
        "ranks' outputs by; a row that reads no key gets 0 and -inf."
    ),
    op_type="dsa_backend_hip",
    axes={
        "n_rows": axis("Query rows: decode tokens or a verify's draft tokens.", 1, 65536, is_var=True),
        # the heads supports_sparse_mla_decode_gfx950 takes: the replaced branch's, as _POOL_ROWS
        "n_heads": axis("Query heads: the DCP group's, gathered on this rank.", 16, 128, multiple_of=16),
        "topk": axis("Entries per row of indices: the global top-k's width.", 1, 65536),
        "n_slots": axis("Rows of this rank's KV pool.", 1, _POOL_ROWS, is_var=True, any_size=True),
    },
    # distinct slots
    constraints=("topk <= n_slots",),
    inputs={
        "q_nope": TensorSpec(
            description="Each row's query per head in the latent's basis: q_nope times W_UK.",
            shape=("n_rows", "n_heads", 512),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "q_rope": TensorSpec(
            description="Each row's RoPE'd query part per head.",
            shape=("n_rows", "n_heads", 64),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "kv_cache": TensorSpec(
            description=(
                "This rank's KV pool, a row a slot in sglang's fp8_ds_mla layout: 512 fp8 e4m3 NoPE "
                "values, an fp32 scale per 128 of them, then 64 bf16 RoPE values."
            ),
            shape=("n_slots", 656),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            distribution=Fp8DsMlaRowsDistribution(),
        ),
        "indices": TensorSpec(
            description=(
                "Each row's slots in this rank's pool, its share of the global top-k first; a row "
                "reads its first topk_length (sglang pads the rest with -1)."
            ),
            shape=("n_rows", "topk"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "topk_length": TensorSpec(
            description="How many of its indices each row reads: this rank's share of its top-k.",
            shape=("n_rows",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="topk"),
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
            shape=("n_rows", "n_heads", 512),
            dtype=_BF16,
            # sglang's kernel rounds q's NoPE part to MXFP8: the bound leaves it that error
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
        "lse": TensorSpec(
            description="Natural log of each head's sum of exp(score).",
            shape=("n_rows", "n_heads"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            # that MXFP8 q moves sglang's LSE by up to about 0.1 on these inputs
            error_bounds=ToleranceBound(absolute_tolerance=0.25, relative_tolerance=0.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (DSA_SPARSE_MLA_DCP_PARTIAL,)
