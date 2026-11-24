# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One of main's contracts re-cut for a host target: the contract, a schema file of it, and the reference
and baseline that stand on main's."""

import json

from aklr.core.contract.baseline import OpBaseline
from aklr.core.contract.op import OpContract
from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import DType, FiniteDistribution, ToleranceBound
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
_CALLS: list[dict] = []  # each call main's reference and baseline below took


class GemmReference(OpReference):
    """Main's reference, as far as the calls it takes."""

    def reference(self, **call):
        _CALLS.append(call)
        return {"out": "expected"}


class GemmBaseline(OpBaseline):
    """Main's baseline, as far as the calls it takes."""

    def baseline(self, **call):
        _CALLS.append(call)
        return {"out": "outputs"}


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
                    ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-3),
                ),
                (None, ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)),
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
    reference=GemmReference(),
    baseline=GemmBaseline(),
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


def test_the_reference_and_the_baseline_are_classes_over_mains_under_the_targets_names() -> None:
    contract = _TUNED.contract(_MAIN)
    scope = {"MainGemmReference": GemmReference, "MainGemmBaseline": GemmBaseline}
    source = _TUNED.class_source(_MAIN, contract, "reference") + _TUNED.class_source(
        _MAIN, contract, "baseline"
    )
    exec(source, scope)  # noqa: S102 -- the source under test
    _CALLS.clear()

    assert scope["GemmA16w16Reference"]().reference(A=1, B=2, bias=None, otype="dtype", out=None) == {
        "out": "expected"
    }
    assert scope["GemmA16w16Baseline"]().baseline(A=1, B=2, bias=None, otype="dtype", out="out") == {
        "out": "outputs"
    }
    assert _CALLS == [
        dict(a=1, b=2, trans_b=True, bias=None, out_dtype="dtype", out=None),
        dict(a=1, b=2, trans_b=True, bias=None, out_dtype="dtype", out="out"),
    ]
    assert isinstance(scope["GemmA16w16Baseline"](), GemmBaseline)  # main's setup, teardown and profile


def test_a_renamed_output_is_named_the_targets_in_the_results() -> None:
    adaptation = Adaptation(
        source="gemm@1.0",
        family="kimi_k3_tiny_gemm",
        target="gemm_a16w16_activation",
        versions=(1,),
        rename={"a": "x", "b": "w", "out_dtype": "dtype", "out": "y"},
        pin={"trans_b": True},
    )
    scope = {"MainGemmReference": GemmReference}
    exec(adaptation.class_source(_MAIN, adaptation.contract(_MAIN), "reference"), scope)  # noqa: S102
    assert scope["GemmA16w16ActivationReference"]().reference(x=1, w=2) == {"y": "expected"}


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
