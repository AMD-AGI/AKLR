# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's TritonAttnBackend.forward_decode on one rank: an absorbed MLA decode."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    ATTENTION_FP8_SNR,
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
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
from aklr.plugins.sglang.layers.mla_decode.baseline import MlaDecodeBaseline
from aklr.plugins.sglang.layers.mla_decode.reference import MlaDecodeReference

__all__ = ["CONTRACTS", "MLA_DECODE"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)

MLA_DECODE = OpContract(
    contract_id="mla_decode@1.0",
    description=(
        "sglang's TritonAttnBackend.forward_decode for an absorbed MLA layer past its KV write, "
        "one query per sequence: each query head attends the slots "
        "kv_indices[kv_indptr[r]:kv_indptr[r + 1]] of the latent cache (page size 1, no RoPE), "
        "scores q . row * sm_scale over all 576 values, and output is the softmax-weighted sum "
        "of the rows' first 512 values"
    ),
    op_type="mla_decode",
    axes={
        "n_seqs": axis("Decode rows: one query each, padding rows included.", 1, 2048, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "n_slots": axis("Rows of the latent cache.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_offsets": axis("Offsets in kv_indptr: n_seqs + 1.", 2, 2049, is_var=True),
        "n_kv_tokens": axis("Entries of kv_indices.", 1, 2**31 - 1, is_var=True, any_size=True),
    },
    constraints=("n_offsets == n_seqs + 1", "n_kv_tokens >= n_seqs"),
    inputs={
        "q": TensorSpec(
            description="Each row's absorbed query per head: 512 latent values, then 64 rope ones.",
            shape=("n_seqs", "n_heads", 576),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "kv_cache": TensorSpec(
            description="This layer's latent cache: 512 latent values then 64 rope ones a slot.",
            shape=("n_slots", 1, 576),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "kv_indptr": TensorSpec(
            description="Where each row's slots start in kv_indices, then where the last row's end.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_kv_tokens"),
        ),
        "kv_indices": TensorSpec(
            description="The cache slots each row attends, row after row.",
            shape=("n_kv_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1"),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: 192**-0.5 for a 128 + 64 query head.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=192**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each head's softmax-weighted sum of the slots' first 512 values.",
            shape=("n_seqs", "n_heads", 512),
            dtype=DTypeSpec(same_as_tensor="q"),
            # sglang's own kernel, which rounds q and the softmax weights to fp8, comes to about
            # 27-29 dB on the schema's workloads
            error_bounds=ATTENTION_FP8_SNR,
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=MlaDecodeReference(),
    baseline=MlaDecodeBaseline(),
)

CONTRACTS = (MLA_DECODE,)
