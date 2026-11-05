# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    ColumnMajor,
    DeterminismLevel,
    Distribution,
    DotProductBound,
    DType,
    E8M0Scales,
    ErrorBound,
    FiniteDistribution,
    LogUniformDistribution,
    OutputRoundingBound,
    Preshuffle,
    ToleranceBound,
)
from aklr.core.contract.spec import (
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
    derived_axis,
)

__all__ = [
    "BMM",
    "CONTRACTS",
    "GEMM",
    "GEMM_A4W4_MXFP4",
    "GEMM_A8W8",
    "GEMM_A8W8_BLOCKWISE_SCALED",
    "GEMM_A8W8_MXFP8",
]

_Q8 = (DType.INT8, DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2)
_FP8 = (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2)
_F16 = (DType.BFLOAT16, DType.FLOAT16)
_F32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))

_FINITE_DISTRIBUTION = FiniteDistribution()
_FLOAT32_SCALE_DISTRIBUTION = LogUniformDistribution(log2_lo="-1", log2_hi="1")
_E8M0_SCALE_DISTRIBUTION = LogUniformDistribution(log2_lo="-8", log2_hi="8")

# prefill MLA kv_b runs over a whole cached prefix in one call: 342,073 rows at AgentX p90
_M = axis("Tokens.", 1, 2**20, is_var=True)
_N = axis("Output columns.", 16, 262144, multiple_of=16)
# lm_head under TP8 of a vocabulary divisible by 64 only: Muse Spark's 202048 / 8 = 25256
_N_A16W16 = axis("Output columns.", 16, 262144, multiple_of=8)
_K = axis("Reduction length.", 128, 65536, multiple_of=128)
_TRANS_B = attr("b is (n, k) when true.", True, False, default=True, param_name="trans_b")
_HAS_BIAS = attr("Whether bias is passed.", False, True, default=False)


def _b(
    description: str,
    dtypes: tuple[DType, ...],
    distribution: Distribution,
    batch: tuple[str, ...] = (),
) -> TensorSpec:
    cases = {"true": (*batch, "n", "k"), "false": (*batch, "k", "n")}
    return TensorSpec(
        description=description,
        shape=ShapeByAttrSpec(attr_name="trans_b", cases=cases),
        dtype=DTypeSpec(any_of=dtypes, same_as_tensor="a"),
        distribution=distribution,
        lifetime="weight",
    )


def _bias(description: str, shape: tuple[str, ...], dtype_spec: DTypeSpec) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype_spec,
        present_when="has_bias",
        distribution=_FINITE_DISTRIBUTION,
    )


def _dot_k(max_split_k: int = 16) -> DotProductBound:
    return DotProductBound(
        length="k",
        accumulate_dtype=DType.FLOAT32,
        max_split_k=max_split_k,
    )


def _out(
    description: str,
    shape: tuple[str, ...],
    any_of: tuple[DType, ...] | None = None,
    default: DType | None = None,
    same_as_tensor: str | None = None,
    error_bounds: tuple[tuple[str | None, ErrorBound], ...] = ((None, _dot_k()),),
) -> dict[str, TensorSpec]:
    dtype = DTypeSpec(any_of=any_of, default=default, same_as_tensor=same_as_tensor)
    out = TensorSpec(description=description, shape=shape, dtype=dtype, error_bounds=error_bounds)
    return {"out": out}


_F16_GEMM_BIAS = DTypeSpec(any_of=(*_F16, DType.FLOAT32))

# A shuffled b is the same maths in a layout no plain kernel reads.
_B_UNSHUFFLED = LayoutFact(location="tensor_attr", name="b.is_shuffled", expected_value=False)
_UNSHUFFLED_VARIANTS = LayoutVariantsSpec(by_name={"row_major": LayoutVariant(detect_facts=(_B_UNSHUFFLED,))})

GEMM = OpContract(
    contract_id="gemm@1.0",
    description="16-bit gemm: out = a @ b.T + bias (b is (k, n) unless trans_b).",
    op_type="gemm",
    axes={"m": _M, "n": _N_A16W16, "k": _K},
    attrs={"trans_b": _TRANS_B, "has_bias": _HAS_BIAS},
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=_F16),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": _b("Weights; (n, k) when trans_b.", _F16, _FINITE_DISTRIBUTION),
        "bias": _bias("Added to every output row.", ("n",), _F16_GEMM_BIAS),
    },
    outputs=_out("a @ b.T + bias.", ("m", "n"), any_of=(*_F16, DType.FLOAT32), same_as_tensor="a"),
    layout_variants=_UNSHUFFLED_VARIANTS,
)

GEMM_A8W8 = OpContract(
    contract_id="gemm_a8w8@1.0",
    description="Per-token 8-bit gemm: out = (a * a_scale) @ (b * b_scale).T + bias.",
    op_type="gemm",
    axes={"m": _M, "n": _N, "k": _K},
    attrs={"trans_b": _TRANS_B, "has_bias": _HAS_BIAS},
    constraints=("b.dtype == a.dtype",),
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=_Q8),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": _b("Weights; (n, k) when trans_b.", _Q8, _FINITE_DISTRIBUTION),
        "a_scale": TensorSpec(
            description="Per row of a.",
            shape=("m", 1),
            dtype=_F32,
            can_squeeze=True,
            distribution=_FLOAT32_SCALE_DISTRIBUTION,
        ),
        "b_scale": TensorSpec(
            description="Per output column.",
            shape=("n", 1),
            dtype=_F32,
            can_squeeze=True,
            distribution=_FLOAT32_SCALE_DISTRIBUTION,
            lifetime="weight",
        ),
        "bias": _bias("Added to every output row.", ("n",), _F32),
    },
    outputs=_out(
        "a @ b.T, scaled.",
        ("m", "n"),
        any_of=_F16,
        default=DType.BFLOAT16,
        error_bounds=(("a.dtype == 'int8'", OutputRoundingBound()), (None, _dot_k())),
    ),
    layout_variants=_UNSHUFFLED_VARIANTS,
)


def _operand_format_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="operand_format", expected_value=value),)


# Fixed 128 x 128 blocks, K-major scales: all aiter passes.
# FIXME(main-kernels): main's compare_gemm_outputs (atol=rtol 1e-3 fp16, 1e-2 bf16, every
# element) and no determinism check; gfx950 fp8 MFMA accumulates below _dot_k()'s IEEE
# fp32 model (aiter is bit-identical) and k8192's split-K order follows CTA arrival.
# FIXME(main-kernels): main draws these scales from U(0.125, 1); 2**U[-3, 0] keeps that
# range, log-uniform. On 2**U[-1, 1] a k1024 fp8 MFMA tail element breaks main's
# tolerance.
_MAIN_SCALE_DISTRIBUTION = LogUniformDistribution(log2_lo="-3", log2_hi="0")
_MAIN_GEMM_TOLERANCE = (
    (
        "out.dtype == 'float16'",
        ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-3),
    ),
    (None, ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)),
)

GEMM_A8W8_BLOCKWISE_SCALED = OpContract(
    contract_id="gemm_a8w8_blockwise_scaled@1.0",
    description=(
        "fp8 gemm dequantized per block: a fp32 scale per 128 x 128 of b,"
        " per 1 x 128 of a, K-major or MN-major. operand_format is"
        " '<a>:<b>'; 'none:preshuffled' is aiter's 16 x 16 shuffle_weight b with a"
        " raw column-major a_scale. Set a_scale.is_row_major to pass a row-major"
        " a_scale with a shuffled b."
    ),
    op_type="gemm",
    axes={
        "m": _M,
        "n": _N,
        "k": _K,
        "scale_rows": derived_axis("Weight-scale rows.", "ceil_div(n, 128)"),
        "scale_columns": derived_axis("Scale columns.", "ceil_div(k, 128)"),
    },
    attrs={
        "scale_major_mode": attr(
            "a_scale is (m, scale_columns) when K, (scale_columns, m) when MN.",
            "K",
            "MN",
            default="K",
            param_name="scale_major_mode",
        ),
        "trans_b": _TRANS_B,
        "has_bias": _HAS_BIAS,
    },
    # aiter's shuffled kernel: NT, K-major, e4m3, no bias.
    constraints=(
        (
            "layout_variant == 'row_major' or (trans_b and scale_major_mode == 'K'"
            " and (not has_bias) and a.dtype == 'float8_e4m3fn')"
        ),
    ),
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=_FP8),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": _b("Weights; (n, k) when trans_b.", _FP8, _FINITE_DISTRIBUTION),
        "a_scale": TensorSpec(
            description="Per 1 x 128 block of a.",
            shape=ShapeByAttrSpec(
                attr_name="scale_major_mode",
                cases={"K": ("m", "scale_columns"), "MN": ("scale_columns", "m")},
            ),
            dtype=_F32,
            distribution=_MAIN_SCALE_DISTRIBUTION,
        ),
        "b_scale": TensorSpec(
            description="Per 128 x 128 block of b.",
            shape=("scale_rows", "scale_columns"),
            dtype=_F32,
            distribution=_MAIN_SCALE_DISTRIBUTION,
            lifetime="weight",
        ),
        "bias": _bias("Added to every output row.", ("n",), _F32),
    },
    outputs=_out(
        "a @ b.T, scaled.",
        ("m", "n"),
        any_of=_F16,
        default=DType.BFLOAT16,
        error_bounds=_MAIN_GEMM_TOLERANCE,
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),  # FIXME(main-kernels)
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=(*_operand_format_facts("none:none"), _B_UNSHUFFLED)),
            "b_shuffled": LayoutVariant(
                layouts={"b": Preshuffle(tile=(16, 16)), "a_scale": ColumnMajor()},
                detect_facts=(
                    *_operand_format_facts("none:preshuffled"),
                    LayoutFact(
                        location="tensor_attr",
                        name="a_scale.is_row_major",
                        expected_value=False,
                    ),
                ),
            ),
            # The same shuffled b with a row-major a_scale.
            "b_shuffled_row_major_a_scale": LayoutVariant(
                layouts={"b": Preshuffle(tile=(16, 16))},
                detect_facts=(
                    *_operand_format_facts("none:preshuffled"),
                    LayoutFact(
                        location="tensor_attr",
                        name="a_scale.is_row_major",
                        expected_value=True,
                    ),
                ),
            ),
        }
    ),
)


GEMM_A8W8_MXFP8 = OpContract(
    contract_id="gemm_a8w8_mxfp8@1.0",
    description="mxfp8 gemm: (a * a_scale) @ (b * b_scale).T + bias, e8m0 scales per 32.",
    op_type="gemm",
    axes={
        "m": _M,
        "n": _N,
        "k": _K,
        "scale_columns": derived_axis("Scale columns.", "k // 32"),
    },
    attrs={
        "trans_b": attr("b is (n, k).", True, default=True, param_name="trans_b"),
        "has_bias": _HAS_BIAS,
    },
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": TensorSpec(
            description="Weights.",
            shape=("n", "k"),
            dtype=DTypeSpec(same_as_tensor="a"),
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "a_scale": TensorSpec(
            description="Per 1 x 32 block of a.",
            shape=("m", "scale_columns"),
            dtype=_E8M0,
            distribution=_E8M0_SCALE_DISTRIBUTION,
        ),
        "b_scale": TensorSpec(
            description="Per 1 x 32 block of b.",
            shape=("n", "scale_columns"),
            dtype=_E8M0,
            distribution=_E8M0_SCALE_DISTRIBUTION,
            lifetime="weight",
        ),
        "bias": _bias("Added to every output row.", ("n",), _F32),
    },
    outputs=_out(
        "a @ b.T, scaled.",
        ("m", "n"),
        any_of=_F16,
        default=DType.BFLOAT16,
        error_bounds=((None, _dot_k(max_split_k=1)),),
    ),
    layout_variants=_UNSHUFFLED_VARIANTS,
)


# aiter's scales come from dynamic_mxfp4_quant(shuffle=True): padded, then shuffled.
_SHUFFLED_SCALES = E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)

# NT only.
GEMM_A4W4_MXFP4 = OpContract(
    contract_id="gemm_a4w4_mxfp4@1.0",
    description=("mxfp4 gemm: alpha * (a * a_scale) @ (b * b_scale).T + bias, scales per 32."),
    op_type="gemm",
    axes={
        "m": _M,
        "n": _N,
        "k": axis("Reduction length.", 64, 65536, multiple_of=64),
        "scale_columns": derived_axis("Scale columns.", "k // 32"),
    },
    attrs={
        "trans_b": attr("b is (n, k).", True, default=True, param_name="trans_b"),
        "has_bias": _HAS_BIAS,
    },
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": TensorSpec(
            description="Weights.",
            shape=("n", "k"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "a_scale": TensorSpec(
            description="Per 1 x 32 block of a.",
            shape=("m", "scale_columns"),
            dtype=_E8M0,
            distribution=_E8M0_SCALE_DISTRIBUTION,
        ),
        "b_scale": TensorSpec(
            description="Per 1 x 32 block of b.",
            shape=("n", "scale_columns"),
            dtype=_E8M0,
            distribution=_E8M0_SCALE_DISTRIBUTION,
            lifetime="weight",
        ),
        "alpha": ScalarSpec(
            description="Product scale.",
            dtype=DType.FLOAT32,
            values=(1.0,),
            default=1.0,
        ),
        "beta": ScalarSpec(
            description="out scale; 0, so out is not read.",
            dtype=DType.FLOAT32,
            values=(0.0,),
            default=0.0,
        ),
        "bias": _bias("Added to every output row.", ("n",), _F32),
    },
    outputs=_out(
        "a @ b.T, scaled.",
        ("m", "n"),
        any_of=_F16,
        default=DType.BFLOAT16,
        error_bounds=((None, _dot_k(max_split_k=1)),),
    ),
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_operand_format_facts("none:none")),
            "b_shuffled": LayoutVariant(
                layouts={
                    "b": Preshuffle(tile=(16, 16)),
                    "a_scale": _SHUFFLED_SCALES,
                    "b_scale": _SHUFFLED_SCALES,
                },
                detect_facts=_operand_format_facts("none:preshuffled"),
            ),
        }
    ),
)

BMM = OpContract(
    contract_id="bmm@1.0",
    description="Batched 16-bit gemm: out[i] = a[i] @ b[i].T + bias[i], over all m.",
    op_type="gemm",
    axes={
        "batch": axis("Independent matmuls.", 1, 65536, is_var=True),
        "m": _M,
        "n": _N,
        "k": _K,
    },
    attrs={"trans_b": _TRANS_B, "has_bias": _HAS_BIAS},
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("batch", "m", "k"),
            dtype=DTypeSpec(any_of=_F16),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "b": _b(
            "Weights; (batch, n, k) when trans_b.",
            _F16,
            _FINITE_DISTRIBUTION,
            batch=("batch",),
        ),
        "bias": _bias("Added to every output row of its batch.", ("batch", "n"), _F16_GEMM_BIAS),
    },
    outputs=_out("a[i] @ b[i].T + bias[i].", ("batch", "m", "n"), same_as_tensor="a"),
    layout_variants=_UNSHUFFLED_VARIANTS,
)

CONTRACTS = (GEMM, GEMM_A8W8, GEMM_A8W8_BLOCKWISE_SCALED, GEMM_A8W8_MXFP8, GEMM_A4W4_MXFP4, BMM)
