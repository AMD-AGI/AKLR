# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contracts written as the python source that builds them."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import ColumnMajor, DotProductBound, DType, FiniteDistribution, Preshuffle
from aklr.core.contract.spec import (
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
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM
from codegen.contract_source import ContractSource

# a contract using what the families' contracts use: attrs, a shape by attr, a scalar, a derived axis,
# branches of bounds, layouts and facts
_GEMM = OpContract(
    contract_id="written_gemm@1.0",
    description='out = a @ b.T, "scaled".',
    op_type="kimi_k3_tiny_gemm",
    axes={
        "m": axis("Rows.", 1, 64, is_var=True),
        "n": axis("Columns.", 16, 64, multiple_of=16),
        "k": axis("Length.", 128, 256),
        "blocks": derived_axis("Scale columns.", "ceil_div(k, 128)"),
    },
    attrs={
        "trans_b": attr("b is (n, k) when true.", True, False, default=True, param_name="trans_b"),
        "has_bias": attr("Whether bias is passed.", False, True, default=False),
    },
    constraints=("trans_b or (not has_bias)",),
    inputs={
        "a": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=FiniteDistribution(),
        ),
        "b": TensorSpec(
            description="Weights.",
            shape=ShapeByAttrSpec(attr_name="trans_b", cases={"true": ("n", "k"), "false": ("k", "n")}),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16), same_as_tensor="a"),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "a_scale": TensorSpec(
            description="Per block of a.",
            shape=("m", "blocks"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=FiniteDistribution(),
        ),
        "bias": TensorSpec(
            description="Added to every row.",
            shape=("n",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            present_when="has_bias",
            distribution=FiniteDistribution(),
        ),
        "alpha": ScalarSpec(description="Product scale.", dtype=DType.FLOAT32, values=(1.0,), default=1.0),
    },
    outputs={
        "out": TensorSpec(
            description="a @ b.T + bias.",
            shape=("m", "n"),
            dtype=DTypeSpec(
                any_of=(DType.BFLOAT16, DType.FLOAT16), default=DType.BFLOAT16, param_name="dtype"
            ),
            error_bounds=(
                (
                    "out.dtype == 'float16'",
                    DotProductBound(length="k", accumulate_dtype=DType.FLOAT32, max_split_k=1),
                ),
                (None, DotProductBound(length="k", accumulate_dtype=DType.FLOAT32, max_split_k=16)),
            ),
        )
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(
                detect_facts=(LayoutFact(location="tensor_attr", name="b.is_shuffled", expected_value=False),)
            ),
            "b_shuffled": LayoutVariant(
                layouts={"b": Preshuffle(tile=(16, 16)), "a_scale": ColumnMajor()},
                detect_facts=(LayoutFact(location="tensor_attr", name="b.is_shuffled", expected_value=True),),
            ),
        }
    ),
)


def test_the_source_builds_the_contracts_it_was_written_from() -> None:
    text = ContractSource({"KIMI": KIMI_K3_TINY_GEMM, "GEMM": _GEMM}).text()
    scope: dict = {}
    exec(text, scope)  # noqa: S102 -- the source under test
    assert scope["CONTRACTS"] == (KIMI_K3_TINY_GEMM, _GEMM)
    assert scope["KIMI"] == KIMI_K3_TINY_GEMM


def test_the_source_reads_as_a_contract_is_written() -> None:
    """Constructor calls of the public names, nothing a default already says, a short call on one line."""
    text = ContractSource({"GEMM": _GEMM}).text()
    assert text.startswith("from aklr.core.contract.op import OpContract\n")
    assert "\nfrom aklr.core.contract.semantics import ColumnMajor, DType, DotProductBound," in text
    assert "._" not in text and "model_validate" not in text
    assert "workloads" not in text and 'kind="tensor"' not in text and 'kind="row_major"' not in text
    assert '\n    contract_id="written_gemm@1.0",\n' in text
    assert "\n    description='out = a @ b.T, \"scaled\".',\n" in text
    assert '\n    constraints=("trans_b or (not has_bias)",),\n' in text
    assert 'LayoutFact(location="tensor_attr", name="b.is_shuffled", expected_value=False)' in text
    assert "distribution=FiniteDistribution()," in text  # one unconditional branch, as written by hand
    assert text.endswith("\nCONTRACTS = (GEMM,)\n")
