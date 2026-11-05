# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-precision MXFP8/BF16 CSR attention over prefix and extend pools."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    CuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    Mxfp8NopeRowsDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import DTypeSpec, LayoutVariant, LayoutVariantsSpec, ScalarSpec, TensorSpec, axis


def _tensor(description, shape, dtype, distribution, *, lifetime="per_call"):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=DTypeSpec(any_of=(dtype,)),
        distribution=distribution,
        lifetime=lifetime,
    )


def _pool(prefix):
    return {
        f"{prefix}_nope": _tensor(
            "Packed NoPE rows: 448 FP8 codes, 14 E8M0 scales, padding to 512 bytes.",
            (f"{prefix}_rows", 512),
            DType.UINT8,
            Mxfp8NopeRowsDistribution(),
            lifetime="weight",
        ),
        f"{prefix}_rope": _tensor(
            "RoPE keys and values corresponding to the NoPE rows.",
            (f"{prefix}_rows", 64),
            DType.BFLOAT16,
            FiniteDistribution(),
            lifetime="weight",
        ),
        f"{prefix}_indices": _tensor(
            "CSR entries selecting rows of the pool.",
            (f"{prefix}_nnz",),
            DType.INT32,
            IntRangeDistribution(lo=0, hi=f"{prefix}_rows - 1"),
        ),
        f"{prefix}_indptr": _tensor(
            "Monotone CSR offsets for each query, beginning with zero.",
            ("n_offsets",),
            DType.INT32,
            CuSeqlensDistribution(min_seqlen=0, total=f"{prefix}_nnz"),
        ),
    }


SPARSE_PREFILL_MXFP8 = OpContract(
    contract_id="sparse_prefill_mxfp8@1.0",
    description=(
        "Sparse attention over two CSR ranges. NoPE uses block-scaled E4M3FN and "
        "RoPE BF16; each head's sink contributes only to the softmax denominator."
    ),
    op_type="mla_prefill",
    axes={
        "n_tokens": axis("Query rows.", 1, 16384, is_var=True),
        "n_heads": axis("Query heads.", 16, 128, multiple_of=16),
        "prefix_rows": axis("Rows of the prefix pool.", 1, 8388608, is_var=True),
        "extend_rows": axis("Rows of the extend pool.", 1, 131072, is_var=True),
        "prefix_nnz": axis("Prefix CSR entries.", 0, 268435456, is_var=True),
        "extend_nnz": axis("Extend CSR entries.", 0, 268435456, is_var=True),
        "n_offsets": axis("CSR offsets, queries plus one.", 2, 16385, is_var=True),
    },
    constraints=("n_offsets == n_tokens + 1",),
    inputs={
        "q_nope": _tensor(
            "Packed NoPE query rows.",
            ("n_tokens", "n_heads", 512),
            DType.UINT8,
            Mxfp8NopeRowsDistribution(),
        ),
        "q_rope": _tensor(
            "Query RoPE components.",
            ("n_tokens", "n_heads", 64),
            DType.BFLOAT16,
            FiniteDistribution(),
        ),
        **_pool("prefix"),
        **_pool("extend"),
        "sinks": _tensor("Per-head softmax sink logits.", ("n_heads",), DType.FLOAT32, FiniteDistribution()),
        "softmax_scale": ScalarSpec(
            description="Multiplier of the combined QK score.",
            dtype=DType.FLOAT32,
            interval=(1e-10, 1.0),
            default=512**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Weighted sum of 448 dequantized NoPE and 64 RoPE values.",
            shape=("n_tokens", "n_heads", 512),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (SPARSE_PREFILL_MXFP8,)
