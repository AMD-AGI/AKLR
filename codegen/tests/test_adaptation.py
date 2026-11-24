# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One of main's contracts re-cut for a host target: the contract, a schema file of it, and the reference
and baseline that stand on main's."""

import json
from types import SimpleNamespace

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DotProductBound, DType, FiniteDistribution
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    PointSpec,
    ShapeByAttrSpec,
    TensorSpec,
    attr,
    axis,
)
from codegen.adaptation import Adaptation

_F16 = (DType.BFLOAT16, DType.FLOAT16)
# a 16-bit gemm as main's aklr/ops writes one: its own names, b either way round, an optional bias
_MAIN = OpContract(
    contract_id="gemm@1.0",
    description="out = a @ b.T + bias.",
    op_type="kimi_k3_tiny_gemm",
    axes={
        "m": axis("Rows.", 1, 64, is_var=True),
        "n": axis("Columns.", 16, 64),
        "k": axis("Length.", 128, 256),
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
            dtype=DTypeSpec(any_of=_F16),
            distribution=FiniteDistribution(),
        ),
        "b": TensorSpec(
            description="Weights.",
            shape=ShapeByAttrSpec(attr_name="trans_b", cases={"true": ("n", "k"), "false": ("k", "n")}),
            dtype=DTypeSpec(any_of=_F16, same_as_tensor="a"),
            distribution=FiniteDistribution(),
        ),
        "bias": TensorSpec(
            description="Added to every row.",
            shape=("n",),
            dtype=DTypeSpec(any_of=_F16),
            present_when="has_bias",
            distribution=FiniteDistribution(),
        ),
    },
    outputs={
        "out": TensorSpec(
            description="a @ b.T + bias.",
            shape=("m", "n"),
            dtype=DTypeSpec(any_of=_F16, same_as_tensor="a", param_name="out_dtype"),
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
            )
        }
    ),
)
# aiter's tuned gemm takes A, B (always (n, k)), bias, otype and out
_TUNED = Adaptation(
    source="gemm@1.0",
    family="kimi_k3_tiny_gemm",
    target="gemm_a16w16",
    versions=(1,),
    rename={"a": "A", "b": "B", "out_dtype": "otype"},
    pin={"trans_b": True},
)
_KERNEL = "def run(*, a, b, trans_b, bias, out_dtype, out):\n    out.copy_(a @ (b.mT if trans_b else b))\n"


def test_the_contract_has_the_targets_params_and_none_of_what_the_target_fixes() -> None:
    contract = _TUNED.contract(_MAIN)
    assert contract.contract_id == "gemm_a16w16@1.0"
    assert [p.name for p in contract.api_params] == ["A", "B", "bias", "otype", "out"]
    assert list(contract.attrs) == ["has_bias"]
    assert contract.inputs["B"].shape == ("n", "k")
    assert contract.inputs["B"].dtype.same_as_tensor == "A"
    assert contract.constraints == ()  # true once trans_b is
    assert contract.outputs["out"].error_bounds[0][0] == "out.dtype == 'float16'"
    assert [f.name for f in contract.layout_variants.by_name["row_major"].detect_facts] == ["B.is_shuffled"]


def test_an_input_the_target_never_passes_is_gone() -> None:
    adaptation = Adaptation(
        source="gemm@1.0", family="kimi_k3_tiny_gemm", target="gemm", versions=(1,), pin={"has_bias": False}
    )
    contract = adaptation.contract(_MAIN)
    assert [p.name for p in contract.api_params] == ["a", "b", "trans_b", "out_dtype", "out"]


def test_an_override_replaces_what_main_says() -> None:
    adaptation = Adaptation(
        source="gemm@1.0",
        family="kimi_k3_tiny_gemm",
        target="gemm_a16w16_activation",
        versions=(1,),
        rename={"a": "x", "b": "w", "out_dtype": "dtype", "out": "y"},
        pin={"trans_b": True},
        override={"outputs": {"y": {"dtype": {"same_as_tensor": None, "default": "bfloat16"}}}},
    )
    dtype = adaptation.contract(_MAIN).outputs["y"].dtype
    assert (dtype.same_as_tensor, dtype.default, dtype.param_name) == (None, DType.BFLOAT16, "dtype")


def test_a_schema_keeps_its_kernel_and_its_measurements_under_the_targets_names() -> None:
    point = PointSpec(
        contract_id="gemm@1.0",
        axes={"m": 4, "n": 16, "k": 128},
        dtypes={"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"},
        attrs={"trans_b": True, "has_bias": False},
        layout_variant_name="row_major",
    )
    measured = {"axes": {"m": 4}, "accuracy": "passed", "speedup": 1.5, "microseconds": 3.0}
    kernel = {
        "language": "flydsl",
        "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
        "entry_point": "kernel.py::run",
        "sources": [{"path": "kernel.py", "content": _KERNEL}],
    }
    main_schema = json.loads(_MAIN.narrow(point, workloads=[measured], kernel=kernel).schema_text())

    schema = _TUNED.schema(main_schema, _MAIN, _TUNED.contract(_MAIN))

    assert schema.name == "gemm_a16w16_n16_k128"
    assert schema.workloads[0].speedup == 1.5
    assert schema.kernel.sources[0].content == (
        "def run(*, A, B, bias, otype, out):\n    trans_b = True\n    out.copy_(A @ (B.mT if trans_b else B))\n"
    )
    assert (
        _TUNED.schema({**main_schema, "attrs": {"trans_b": False, "has_bias": False}}, _MAIN, schema) is None
    )


def test_the_reference_and_the_baseline_are_mains_under_the_targets_names() -> None:
    calls = []

    def gemm_reference(point: SimpleNamespace, row_major_args: dict) -> str:
        calls.append((point.attrs, point.dtypes, row_major_args))
        return "reference"

    def gemm_baseline(a, b, trans_b=True, bias=None, out_dtype=None, out=None) -> str:  # noqa: ANN001
        calls.append((a, b, trans_b, bias, out_dtype, out))
        return "baseline"

    scope = {
        "main_gemm_reference": gemm_reference,
        "main_gemm_baseline": gemm_baseline,
        "SimpleNamespace": SimpleNamespace,
    }
    contract = _TUNED.contract(_MAIN)
    exec(_TUNED.reference_source(_MAIN, contract) + _TUNED.baseline_source(_MAIN, contract), scope)  # noqa: S102

    point = SimpleNamespace(
        axes={"m": 4},
        attrs={"has_bias": False},
        dtypes={"A": "bfloat16", "B": "bfloat16", "bias": None, "out": "bfloat16"},
    )
    assert scope["gemm_a16w16_reference"](point, {"A": 1, "B": 2, "bias": None}) == "reference"
    assert scope["gemm_a16w16_baseline"](1, 2, otype="dtype", out="out") == "baseline"
    assert calls == [
        (
            {"trans_b": True, "has_bias": False},
            {"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"},
            {"a": 1, "b": 2, "bias": None},
        ),
        (1, 2, True, None, "dtype", "out"),
    ]


def test_a_description_names_the_targets_params_in_its_formulas_only() -> None:
    """A name of main's that is also a word of the sentence stays the word."""
    main = OpContract(
        **{
            **{name: getattr(_MAIN, name) for name in OpContract.model_fields},
            "description": "out = a @ b.T + bias (b is (n, k)); the out rows are a sum over k of a.",
        }
    )
    adaptation = Adaptation(
        source="gemm@1.0",
        family="kimi_k3_tiny_gemm",
        target="gemm_a16w16_activation",
        versions=(1,),
        rename={"a": "x", "b": "w", "out": "y"},
    )
    assert adaptation.contract(main).description == (
        "y = x @ w.T + bias (w is (n, k)); the out rows are a sum over k of x."
    )
