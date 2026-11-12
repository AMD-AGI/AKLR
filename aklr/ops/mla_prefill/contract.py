# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-precision MXFP8/BF16 CSR attention over prefix and extend pools."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    Mxfp8NopeRowsDistribution,
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
    attr,
    axis,
)


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

_ROWS_LIMIT = 2**20  # one prefill's rows: sglang attends at most 131072 keys at once
_BF16_OR_SAME = DTypeSpec(same_as_tensor="q")

MLA_PREFILL_VARLEN = OpContract(
    contract_id="mla_prefill_varlen@1.0",
    description=(
        "MLA's prefill attention once kv_b has expanded the latent keys, every head on its own: sequence s's "
        "queries are q's rows cu_seqlens_q[s]:cu_seqlens_q[s + 1], its keys and values the rows "
        "cu_seqlens_k[s]:cu_seqlens_k[s + 1] of k and v (with no prefix, the queries' own rows). Causal, aligned "
        "at the end: query i of a sequence of Lq queries over Lk keys reads keys 0 .. Lk - Lq + i. Logits "
        "q . k * sm_scale over qk_head_dim; output the softmax-weighted sum of v. With qkv_fp8 a kernel may "
        "round q, k and v to float8_e4m3fn first, as aiter's fp8 prefill does."
    ),
    op_type="mla_prefill",
    axes={
        "n_q": axis("Query rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True),
        "n_kv": axis(
            "Key and value rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True
        ),
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 4097, is_var=True, any_size=True),
        "n_heads": axis("Heads on this rank; k and v carry q's.", 1, 128),
        "qk_head_dim": axis("Values per q and k head: the no-rope and rope parts.", 64, 256, multiple_of=64),
        "v_head_dim": axis("Values per v and output head.", 64, 256, multiple_of=64),
    },
    attrs={
        "has_prefix": attr(
            "Whether cu_seqlens_k is passed: the sequences' keys start with a cached prefix.",
            False,
            True,
            default=False,
            abbrev="prefix",
        ),
        "qkv_fp8": attr(
            "Whether a kernel may round q, k and v to float8_e4m3fn, scale 1, before it attends.",
            True,
            False,
            default=True,
            param_name="qkv_fp8",
            abbrev="fp8",
        ),
    },
    constraints=(
        "n_offsets - 1 <= n_q",
        "has_prefix or n_kv == n_q",
        # each sequence's keys are at least all the queries: none of its queries reads no key
        "(not has_prefix) or n_kv >= (n_offsets - 1) * n_q",
    ),
    inputs={
        "q": TensorSpec(
            description="Queries.",
            shape=("n_q", "n_heads", "qk_head_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
        "k": TensorSpec(
            description="Keys.",
            shape=("n_kv", "n_heads", "qk_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "v": TensorSpec(
            description="Values.",
            shape=("n_kv", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "cu_seqlens_q": TensorSpec(
            description="Where each sequence's queries start in q, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_q"),
        ),
        "cu_seqlens_k": TensorSpec(
            description="Where each sequence's keys start in k and v, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            present_when="has_prefix",
            distribution=CuSeqlensDistribution(min_seqlen="n_q", total="n_kv"),
        ),
        # always passed: present_when gives the api default None, which declines
        "sm_scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value),
            # MLA's, 1/sqrt(128 + 64): at 1, N(0, 1) logits over 192 values reach +-60, past aiter's fp8 prefill
            default=192**-0.5,
            present_when="True",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="The softmax-weighted sum of the values each query reads.",
            shape=("n_q", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            # rounding q, k and v to fp8 leaves sglang's own path about 26 dB from the exact result
            error_bounds=(
                ("qkv_fp8", SqnrBound(min_sqnr_db=20.0)),
                (None, ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)),
            ),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (MLA_PREFILL_VARLEN, SPARSE_PREFILL_MXFP8)
