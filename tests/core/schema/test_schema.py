# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import aklr.ops
from aklr.core.schema import (
    Build,
    Kernel,
    Schema,
    SchemaCollection,
    SourceFile,
    Tensor,
    Var,
    Workload,
    load_json_file,
    save_json_file,
)
from aklr.core.utils import dtype_str_to_torch_dtype, torch_dtype_to_dtype_str
from tests.op_case import OpCase

_COMMITTED = sorted(Path(aklr.ops.__file__).parent.glob("*/impls/*/*/*.json"))
_DTYPES = (
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float8_e4m3fn", torch.float8_e4m3fn),
    ("float8_e5m2", torch.float8_e5m2),
    ("float8_e4m3fnuz", torch.float8_e4m3fnuz),
    ("float8_e5m2fnuz", torch.float8_e5m2fnuz),
    ("float8_e8m0fnu", torch.float8_e8m0fnu),
    ("float4_e2m1fn_x2", torch.float4_e2m1fn_x2),
    ("int64", torch.int64),
    ("int32", torch.int32),
    ("int16", torch.int16),
    ("int8", torch.int8),
    ("uint8", torch.uint8),
    ("bool", torch.bool),
)


def _kernel(
    *,
    dps: bool = True,
    path: str = "kernel.py",
    entry_point: str = "kernel.py::run",
    target: str = "gfx950",
) -> Kernel:
    return Kernel(
        name="kernel",
        author="tests",
        build=Build(
            language="triton",
            target_hardware=[target],
            entry_point=entry_point,
            destination_passing_style=dps,
        ),
        sources=[SourceFile(path=path, content="def run(*args):\n    pass\n")],
    )


class TestSchemaFile:
    @pytest.mark.parametrize("path", _COMMITTED, ids=lambda p: p.parent.name)
    def test_committed_schema_round_trips(self, path: Path, tmp_path: Path) -> None:
        save_json_file(load_json_file(Schema, path), tmp_path / "schema.json")
        assert (tmp_path / "schema.json").read_text() == path.read_text()

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"outputs": {"a": Tensor(["m"])}}, id="input_output_overlap"),
            pytest.param({"inputs": {"a": Tensor(["n"])}}, id="undefined_axis"),
            pytest.param({"constraints": ["m >"]}, id="constraint_syntax"),
        ],
    )
    def test_rejects(self, change: dict) -> None:
        fields = {
            "op_type": "op",
            "axes": {"m": Var()},
            "inputs": {"a": Tensor(["m"])},
            "outputs": {"out": Tensor(["m"])},
        }
        with pytest.raises(ValidationError):
            Schema(**(fields | change))


class TestTensorDtype:
    @pytest.mark.parametrize(("dtype", "torch_dtype"), _DTYPES, ids=lambda d: str(d))
    def test_dtype_is_the_torch_dtype_of_its_name(self, dtype: str, torch_dtype: torch.dtype) -> None:
        tensor = Tensor(["m"], dtype=dtype)
        assert dtype_str_to_torch_dtype(tensor.dtype) == torch_dtype
        assert torch_dtype_to_dtype_str(torch_dtype) == dtype

    @pytest.mark.parametrize("dtype", ["float4_e2m1", "float8_e8m0", "float64", "unknown"])
    def test_rejects_a_name_outside_the_dtypes(self, dtype: str) -> None:
        with pytest.raises(ValidationError):
            Tensor(["m"], dtype=dtype)


class TestForCall:
    def test_names_the_dtypes_of_the_call(self) -> None:
        case = OpCase.of("fused_moe_gate_up_interleaved")
        schema = case.template.for_call(case.call())
        assert {n: schema.inputs[n].dtype for n in ("w1", "w2", "w1_scale")} == {
            "w1": "float4_e2m1fn_x2",
            "w2": "float4_e2m1fn_x2",
            "w1_scale": "float8_e8m0fnu",
        }

    def test_absent_optional_input_has_no_dtype(self) -> None:
        schema = OpCase.of("gemm_bf16_nt").schema()
        assert schema.name == "gemm_bf16_nt_n32_k64"
        assert schema.inputs["bias"].dtype is None


class TestKernel:
    def test_calling_convention_is_part_of_its_identity(self) -> None:
        dps, value_returning = _kernel(dps=True), _kernel(dps=False)
        assert dps != value_returning
        assert dps.hash() != value_returning.hash()

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"path": "/kernel.py"}, id="absolute_path"),
            pytest.param({"path": "../kernel.py"}, id="parent_path"),
            pytest.param({"entry_point": "other.py::run"}, id="entry_file_missing"),
            pytest.param({"entry_point": "kernel.py"}, id="entry_without_symbol"),
            pytest.param({"target": "gfx950/MI355X/x"}, id="target_hardware"),
        ],
    )
    def test_rejects(self, change: dict) -> None:
        with pytest.raises(ValidationError):
            _kernel(**change)


class TestSchemaCollection:
    def test_add_workloads_keeps_one_per_axes(self, tmp_path: Path) -> None:
        schema = OpCase.of("gemm_bf16_nt").schema()
        SchemaCollection.from_path(str(tmp_path)).add_workloads(
            [(schema, Workload(axes={"M": m})) for m in (4, 4, 8)]
        )
        collection = SchemaCollection.from_path(str(tmp_path))
        assert [s.workloads for s in collection.schemas[schema.name]] == [
            [Workload(axes={"M": 4}), Workload(axes={"M": 8})]
        ]

    def test_same_name_files_must_agree(self, tmp_path: Path) -> None:
        schema = OpCase.of("gemm_bf16_nt").schema()
        collection = SchemaCollection.from_path(str(tmp_path))
        collection.add_workloads([(schema, Workload(axes={"M": 4}))])
        other = Schema(
            name=schema.name,
            op_type=schema.op_type,
            axes=schema.axes,
            inputs=schema.inputs,
            outputs=schema.outputs,
            description="another contract",
        )
        save_json_file(other, collection.layout.path(other))
        with pytest.raises(ValueError, match=schema.name):
            SchemaCollection.from_path(str(tmp_path))
