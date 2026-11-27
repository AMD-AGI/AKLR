# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``OpContract`` taking a host function's params: an output of any name, its dtype picked through a
param of any name."""

from typing import Any

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import DType, tolerance
from aklr.core.contract.spec import DTypeSpec, PointSpec, TensorSpec
from tests.op_case import OpCase

_GEMM = ContractRegistry.default().get("gemm@1.0")
_SAME_AS_A = DTypeSpec(same_as_tensor="a")
# the caller picks the output's dtype through ``otype``, as aiter's gemm_a16w16 takes it
_BY_PARAM = DTypeSpec(
    any_of=(DType.BFLOAT16, DType.FLOAT16, DType.FLOAT32), default=DType.BFLOAT16, param_name="otype"
)
_Y = TensorSpec(
    description="a @ b.T.",
    shape=("m", "n"),
    dtype=_BY_PARAM,
    error_bounds=tolerance(DType.BFLOAT16, DType.FLOAT16),
)
_Y_SAME_AS_A = _Y.model_copy(update={"dtype": _SAME_AS_A})


def _contract(**fields: Any) -> OpContract:
    """gemm@1.0 with ``fields`` replaced, under another id."""
    base = {name: getattr(_GEMM, name) for name in OpContract.model_fields}
    return OpContract(**{**base, "contract_id": "dtype_param@1.0", **fields})


def _point(y_dtype: str) -> PointSpec:
    return PointSpec(
        contract_id="dtype_param@1.0",
        axes={"m": 4, "n": 16, "k": 128},
        dtypes={"a": "bfloat16", "b": "bfloat16", "bias": None, "y": y_dtype},
        attrs={"trans_b": True, "has_bias": False},
        layout_variant_name="row_major",
    )


def _call(otype: torch.dtype | None) -> dict[str, Any]:
    return {
        "a": torch.zeros(4, 128, dtype=torch.bfloat16),
        "b": torch.zeros(16, 128, dtype=torch.bfloat16),
        "trans_b": True,
        "bias": None,
        "otype": otype,
        "y": None,
    }


def test_an_output_of_any_name_takes_its_dtype_param_before_the_allocated_outputs() -> None:
    contract = _contract(outputs={"y": _Y})
    assert [p.name for p in contract.api_params][-2:] == ["otype", "y"]


def test_an_output_without_a_dtype_param_adds_no_param() -> None:
    """``out`` is a name like any other: no ``out_dtype`` comes with it."""
    contract = _contract(outputs={"out": _Y_SAME_AS_A})
    assert "out_dtype" not in [p.name for p in contract.api_params]


def test_out_stands_beside_another_output() -> None:
    contract = _contract(outputs={"out": _Y_SAME_AS_A, "doubled": _Y_SAME_AS_A})
    assert [p.name for p in contract.api_params][-2:] == ["out", "doubled"]


@pytest.mark.parametrize(
    ("passed", "y_dtype"),
    [(torch.float32, "float32"), (torch.float16, "float16"), (None, "bfloat16")],
    ids=["float32", "float16", "its_default"],
)
def test_fast_key_reads_the_output_dtype_from_its_param(passed: torch.dtype | None, y_dtype: str) -> None:
    codec = FastKeyCodec.from_contract(_contract(outputs={"y": _Y}))
    fast_key = FastKeyFnGenerator(codec, None).generate()
    assert codec.to_point(fast_key(_call(passed))) == _point(y_dtype)


def test_fast_key_declines_a_dtype_the_output_does_not_take() -> None:
    fast_key = FastKeyFnGenerator(FastKeyCodec.from_contract(_contract(outputs={"y": _Y})), None).generate()
    assert fast_key(_call(torch.int32)) is None


def test_a_bench_call_passes_the_points_output_dtype_through_its_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(outputs={"y": _Y})
    registry = ContractRegistry([*ContractRegistry.default().contracts(), contract])
    monkeypatch.setattr(ContractRegistry, "default", classmethod(lambda cls: registry))
    case = OpCase(_point("float32"))
    assert case.call(case.inputs())["otype"] == torch.float32


def test_only_an_output_takes_a_dtype_param() -> None:
    a = _GEMM.inputs["a"]
    by_param = a.dtype.model_copy(update={"param_name": "a_dtype"})
    with pytest.raises(ValueError, match="only an output takes a dtype param"):
        _contract(inputs={**_GEMM.inputs, "a": a.model_copy(update={"dtype": by_param})})


def _schema(contract: OpContract, run: str) -> OpContract:
    point = _point("float32").model_copy(
        update={"dtypes": {"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"}}
    )
    return contract.narrow(
        point,
        workloads=[contract.workload_for(point, {})],
        kernel={
            "language": "flydsl",
            "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
            "requires": {"flydsl": ">=0.5"},
            "entry_point": "kernel.py::run",
            "sources": [{"path": "kernel.py", "content": f'"""K."""\n\n\n{run}\n'}],
        },
    )


def test_a_param_the_result_does_not_depend_on_is_the_apis_and_not_the_kernels() -> None:
    """aiter's ``large_m_splitk`` picks aiter's schedule: the api takes it at any value, the Point and
    the kernel entry do not."""
    contract = _contract(ignored_params=("large_m_splitk",))
    params = {p.name: p for p in contract.api_params}
    assert not params["large_m_splitk"].is_required
    schema = _schema(contract, "def run(*, a, b, trans_b, bias, out_dtype, out): ...")
    assert schema.kernel_call_keys == ("a", "b", "trans_b", "bias", "out_dtype", "out")
    fast_key = FastKeyFnGenerator(FastKeyCodec.from_contract(contract), None).generate()
    call = {**_call(None), "out_dtype": None, "out": None}
    del call["otype"], call["y"]
    assert fast_key({**call, "large_m_splitk": True}) == fast_key({**call, "large_m_splitk": False})


def test_an_ignored_param_has_a_name_of_its_own() -> None:
    with pytest.raises(ValueError, match="repeated"):
        _contract(ignored_params=("trans_b",))
