# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``OpContract`` against a host function: the dtype an output takes from a param of any name, and the
params a host function must have."""

from typing import Any

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import DType, Preshuffle, tolerance
from aklr.core.contract.spec import DTypeSpec, LayoutVariant, LayoutVariantsSpec, PointSpec, TensorSpec
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM
from tests.op_case import OpCase

_SAME_AS_X = DTypeSpec(same_as_tensor="x")
# the caller picks the output's dtype through ``dtype``, as aiter's gemms take it
_BY_PARAM = DTypeSpec(
    any_of=(DType.BFLOAT16, DType.FLOAT16, DType.FLOAT32), default=DType.BFLOAT16, param_name="dtype"
)


# an output whose dtype the caller picks; ``model_copy`` gives it another dtype spec or shape
_Y = TensorSpec(
    description="x @ w.T.",
    shape=("m", "n"),
    dtype=_BY_PARAM,
    error_bounds=tolerance(DType.BFLOAT16, DType.FLOAT16),
)
_Y_SAME_AS_X = _Y.model_copy(update={"dtype": _SAME_AS_X})


def _contract(**fields: Any) -> OpContract:
    """Kimi-K3's tiny gemm contract with ``fields`` replaced, under another id."""
    base = {name: getattr(KIMI_K3_TINY_GEMM, name) for name in OpContract.model_fields}
    return OpContract(**{**base, "contract_id": "dtype_param@1.0", **fields})


def _point(y_dtype: str) -> PointSpec:
    return PointSpec(
        contract_id="dtype_param@1.0",
        axes={"m": 4, "n": 16, "k": 128},
        dtypes={"x": "bfloat16", "w": "bfloat16", "y": y_dtype},
        attrs={},
        layout_variant_name="row_major",
    )


def test_an_output_of_any_name_takes_its_dtype_param_before_the_allocated_outputs() -> None:
    contract = _contract(outputs={"y": _Y})
    assert [p.name for p in contract.api_params] == ["x", "w", "dtype", "y"]


def test_an_output_without_a_dtype_param_adds_no_param() -> None:
    """``out`` is a name like any other: no ``out_dtype`` comes with it."""
    assert [p.name for p in KIMI_K3_TINY_GEMM.api_params] == ["x", "w", "out"]


def test_out_stands_beside_another_output() -> None:
    contract = _contract(outputs={"out": _Y_SAME_AS_X, "doubled": _Y_SAME_AS_X})
    assert [p.name for p in contract.api_params] == ["x", "w", "out", "doubled"]


@pytest.mark.parametrize(
    ("passed", "y_dtype"),
    [(torch.float32, "float32"), (torch.float16, "float16"), (None, "bfloat16")],
    ids=["float32", "float16", "its_default"],
)
def test_fast_key_reads_the_output_dtype_from_its_param(passed: torch.dtype | None, y_dtype: str) -> None:
    contract = _contract(outputs={"y": _Y})
    codec = FastKeyCodec.from_contract(contract)
    fast_key = FastKeyFnGenerator(codec, None).generate()
    call = {
        "x": torch.zeros(4, 128, dtype=torch.bfloat16),
        "w": torch.zeros(16, 128, dtype=torch.bfloat16),
        "dtype": passed,
        "y": None,
    }
    assert codec.to_point(fast_key(call)) == _point(y_dtype)


def test_fast_key_declines_a_dtype_the_output_does_not_take() -> None:
    contract = _contract(outputs={"y": _Y})
    fast_key = FastKeyFnGenerator(FastKeyCodec.from_contract(contract), None).generate()
    call = {
        "x": torch.zeros(4, 128, dtype=torch.bfloat16),
        "w": torch.zeros(16, 128, dtype=torch.bfloat16),
        "dtype": torch.int32,
        "y": None,
    }
    assert fast_key(call) is None


def test_a_schema_keeps_the_dtype_param_among_its_kernels_call_keys() -> None:
    """The kernel entry takes it, as dispatch passes it."""
    contract = _contract(outputs={"y": _Y})
    point = _point("float32")
    schema = contract.narrow(
        point,
        workloads=[contract.workload_for(point, {})],
        kernel={
            "language": "flydsl",
            "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
            "entry_point": "kernel.py::run",
            "sources": [{"path": "kernel.py", "content": "def run(*, x, w, dtype, y): ...\n"}],
        },
    )
    assert schema.kernel_call_keys == ("x", "w", "dtype", "y")


def test_a_bench_call_passes_the_points_output_dtype_through_its_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(outputs={"y": _Y})
    registry = ContractRegistry([*ContractRegistry.default().contracts(), contract])
    monkeypatch.setattr(ContractRegistry, "default", classmethod(lambda cls: registry))
    case = OpCase(_point("float32"))
    assert case.call(case.inputs())["dtype"] == torch.float32


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        (
            {
                "outputs": {
                    "y": _Y.model_copy(update={"dtype": _BY_PARAM.model_copy(update={"param_name": "x"})})
                }
            },
            "repeated",
        ),
        (
            {
                "outputs": {
                    "y": _Y.model_copy(update={"dtype": _BY_PARAM.model_copy(update={"param_name": "m"})})
                }
            },
            "repeated",
        ),
        ({"outputs": {"y": _Y, "z": _Y}}, "repeated"),
        (
            {
                "inputs": {
                    **KIMI_K3_TINY_GEMM.inputs,
                    "w": KIMI_K3_TINY_GEMM.inputs["w"].model_copy(
                        update={"dtype": DTypeSpec(same_as_tensor="x", param_name="dtype")}
                    ),
                }
            },
            "w: only an output takes a dtype param",
        ),
        (
            {
                "outputs": {
                    "y": _Y.model_copy(update={"dtype": DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16))})
                }
            },
            "y: an output without a dtype param has one dtype or the same_as_tensor of an input",
        ),
        (
            {
                "axes": {**KIMI_K3_TINY_GEMM.axes, "j": KIMI_K3_TINY_GEMM.axes["n"]},
                "outputs": {"y": _Y.model_copy(update={"shape": ("m", "j")})},
            },
            "axis 'j' is not readable",
        ),
    ],
    ids=[
        "param_named_as_an_input",
        "param_named_as_an_axis",
        "two_outputs_one_param",
        "param_on_an_input",
        "many_dtypes_and_no_param",
        "allocated_output_alone_carries_an_axis",
    ],
)
def test_a_wrong_dtype_param_is_rejected(fields: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _contract(**fields)


def _host_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def _defaulted_extra(x, w, mode="exact"):
    return x @ w.mT


def _var_args(x, w, *args):
    return x @ w.mT


def _missing_w(x):
    return x


def _undefaulted_extra(x, w, mode):
    return x @ w.mT


def _defaulted_required(x, w=1):
    return x


@pytest.mark.parametrize("fn", [_host_gemm, _defaulted_extra], ids=lambda fn: fn.__name__)
def test_a_host_function_of_any_name_and_body_passes_the_signature_check(fn: Any) -> None:
    KIMI_K3_TINY_GEMM.check_api_signature(fn)


@pytest.mark.parametrize(
    ("fn", "error"),
    [
        (_var_args, r"no positional-only or \*args"),
        (_missing_w, "missing 'w'"),
        (_undefaulted_extra, "need a default"),
        (_defaulted_required, "required 'w' may default only to None"),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_a_host_function_with_other_params_fails_the_signature_check(fn: Any, error: str) -> None:
    with pytest.raises(TypeError, match=error):
        KIMI_K3_TINY_GEMM.check_api_signature(fn)


def test_a_contract_may_have_one_layout_that_is_not_row_major() -> None:
    """A host function that reads its weight in one stored layout only, as aiter's gemms on a shuffled
    weight do: the contract's one variant lists that layout, needs no fact to be told by, and is left
    out of a schema's name."""
    shuffled = LayoutVariantsSpec(
        by_name={"w_shuffled": LayoutVariant(layouts={"w": Preshuffle(tile=(16, 16))})}
    )
    contract = _contract(layout_variants=shuffled)
    point = PointSpec(
        contract_id="dtype_param@1.0",
        axes={"m": 4, "n": 16, "k": 128},
        dtypes={"x": "bfloat16", "w": "bfloat16", "out": "bfloat16"},
        attrs={},
        layout_variant_name="w_shuffled",
    )
    assert contract.name_at(point) == "dtype_param_n16_k128"
