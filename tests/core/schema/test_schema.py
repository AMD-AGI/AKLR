# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import re
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

import aklr.ops
from aklr.core.platform import toolchain
from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch import OpVariants
from aklr.core.schema import (
    Build,
    Const,
    Evaluation,
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
from aklr.ops.fused_moe import fused_moe
from tests.op_sample import GEMM_SAMPLE, OpSample

_COMMITTED = sorted(Path(aklr.ops.__file__).parent.glob("*/impls/*/*/*.json"))
# the ops' own: tests add apis of their toy ops to the declared ones
_VARIANTS = tuple(
    s
    for api in OpVariants.ops().values()
    for s in api.variants
    if s.standard.reference.__module__.startswith("aklr.ops.")
)
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
    requires: str = ">=3.8",
) -> Kernel:
    return Kernel(
        name="kernel",
        author="tests",
        build=Build(
            language="triton",
            target_hardware=[target],
            entry_point=entry_point,
            requires={"triton": requires},
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


@pytest.mark.parametrize(
    ("axes", "error"),
    [
        ({"m": 4, "width": 8}, "non-variable axes ['width']"),
        ({"m": 4, "typo": 8}, "non-variable axes ['typo']"),
    ],
    ids=["constant_override", "unknown_axis"],
)
def test_schema_rejects_invalid_workload_axes(axes: dict[str, int], error: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(error)):
        _schema(axes={"m": Var(), "width": Const(value=8)}, workloads=[Workload(axes=axes)])


def test_a_recorded_workload_may_lack_an_axis_only_outputs_have() -> None:
    # tracing reads a call's axes off its inputs: the audit, not the schema, asks for the outputs'
    schema = _schema(
        axes={"m": Var(), "n": Var()},
        inputs={"a": Tensor(["m"])},
        outputs={"out": Tensor(["m", "n"])},
        workloads=[Workload(axes={"m": 4})],
    )
    assert schema.workloads[0].axes == {"m": 4}


def test_workload_can_omit_axis_used_only_by_optional_input() -> None:
    schema = _schema(
        axes={"m": Var(), "bias_len": Var(), "width": Const(value=8)},
        inputs={"a": Tensor(["m"]), "bias": Tensor(["bias_len"], optional=True)},
        outputs={"out": Tensor(["m", "width"])},
        workloads=[Workload(axes={"m": 4})],
    )
    assert schema.workloads[0].axes == {"m": 4}


def _schema(**fields) -> Schema:
    return Schema(
        **{
            "op_type": "op",
            "axes": {"m": Var()},
            "inputs": {"a": Tensor(["m"])},
            "outputs": {"out": Tensor(["m"])},
        }
        | fields
    )


class TestVariant:
    def test_a_schema_without_one_is_its_ops_default_variant(self) -> None:
        schema = _schema()
        assert (schema.variant_name, schema.op_name, schema.name) == ("default", "op.default", "op")
        assert schema.matches({"a": torch.ones(1)})

    def test_names_the_variant(self) -> None:
        schema = _schema(variant={"wide": "flag"})
        assert (schema.variant_name, schema.op_name, schema.name) == ("wide", "op.wide", "op_wide")

    def test_a_default_with_a_condition_keeps_the_ops_name(self) -> None:
        assert _schema(variant={"default": "not flag"}).name == "op"

    @pytest.mark.parametrize(
        ("condition", "call", "matches"),
        [
            pytest.param("flag", {"flag": True}, True, id="scalar"),
            pytest.param("flag", {"flag": False}, False, id="scalar_unmet"),
            pytest.param("mode == 'k' and not flag", {"mode": "k", "flag": False}, True, id="two_arguments"),
            pytest.param("a.dtype == torch.int8", {"a": torch.ones(2, dtype=torch.int8)}, True, id="dtype"),
            pytest.param("a.layout == torch.sparse_coo", {"a": torch.ones(2)}, False, id="layout"),
            pytest.param(
                "a.shape[0] > b.shape[0]", {"a": torch.ones(3), "b": torch.ones(2)}, True, id="pair"
            ),
        ],
    )
    def test_matches_the_calls_its_condition_holds_on(
        self, condition: str, call: dict, matches: bool
    ) -> None:
        assert _schema(variant={"wide": condition}).matches(call) is matches

    def test_is_kept_in_the_json(self, tmp_path: Path) -> None:
        schema = _schema(variant={"wide": "flag"})
        save_json_file(schema, tmp_path / "schema.json")
        assert json.loads((tmp_path / "schema.json").read_text())["variant"] == {"wide": "flag"}
        assert load_json_file(Schema, tmp_path / "schema.json").op_name == "op.wide"

    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param({"wide": "flag", "narrow": "not flag"}, id="two_variants"),
            pytest.param({"wide": "flag ="}, id="condition_syntax"),
            pytest.param({"Wide One": "flag"}, id="name"),
        ],
    )
    def test_rejects(self, variant: dict) -> None:
        with pytest.raises(ValidationError):
            _schema(variant=variant)

    @pytest.mark.parametrize("template", _VARIANTS, ids=lambda t: t.schema_name({}))
    def test_every_ops_names_fit_a_file_name(self, template: Schema) -> None:
        name = template.schema_name({n: 2**31 for n, a in template.axes.items() if isinstance(a, Const)})
        assert len(name.encode()) <= 255
        assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name)


def test_an_evaluations_kernel_time_is_positive() -> None:
    with pytest.raises(ValidationError, match="microseconds"):
        Evaluation(hardware="gfx950/MI355X", accuracy="passed", microseconds=0)
    assert Evaluation(hardware="gfx950/MI355X", accuracy="passed", microseconds=12.3).microseconds == 12.3


class TestServes:
    _TARGET = DeviceTarget.from_names("gfx950", "MI355X")

    @pytest.mark.parametrize(
        ("kernel", "accuracy", "serves"),
        [
            pytest.param(_kernel(), "passed", True, id="passed_on_the_target"),
            pytest.param(_kernel(), "failed", False, id="failed"),
            pytest.param(_kernel(target="gfx942"), "passed", False, id="kernel_of_another_target"),
            pytest.param(_kernel(requires=">=3.9"), "passed", False, id="kernel_of_a_newer_toolchain"),
            pytest.param(None, "passed", False, id="no_kernel"),
        ],
    )
    def test_serves_a_qualified_workload_by_a_kernel_of_the_target(
        self, monkeypatch: pytest.MonkeyPatch, kernel: Kernel | None, accuracy: str, serves: bool
    ) -> None:
        monkeypatch.setattr(toolchain, "toolchain_version", lambda name: "3.8.0")
        workload = Workload(
            axes={"m": 4}, evaluations=[Evaluation(hardware="gfx950/MI355X", accuracy=accuracy)]
        )
        schema = _schema(kernel=kernel, workloads=[workload])
        assert schema.serves(workload, self._TARGET, 1.03, "") is serves


class TestStandard:
    def test_the_json_has_no_standard(self, tmp_path: Path) -> None:
        schema = GEMM_SAMPLE.schema()
        assert schema.standard is not None
        save_json_file(schema, tmp_path / "schema.json")
        assert (
            not {"standard", "reference", "check", "init"}
            & json.loads((tmp_path / "schema.json").read_text()).keys()
        )

    def test_the_standard_is_not_part_of_the_uuid(self) -> None:
        schema = GEMM_SAMPLE.schema()
        loosened = replace(schema, standard=replace(schema.standard, check=lambda *_: True))
        assert loosened.uuid == schema.uuid

    def test_rejects_a_json_with_a_function(self, tmp_path: Path) -> None:
        schema = GEMM_SAMPLE.schema()
        save_json_file(schema, tmp_path / "schema.json")
        fields = json.loads((tmp_path / "schema.json").read_text()) | {"reference": "def run():\n    pass\n"}
        with pytest.raises(ValidationError):
            TypeAdapter(Schema).validate_python(fields)

    def test_a_schema_read_as_plain_json_has_no_standard(self) -> None:
        assert load_json_file(Schema, _COMMITTED[0]).standard is None

    def test_a_schema_built_in_code_keeps_the_standard_it_is_given(self) -> None:
        assert replace(GEMM_SAMPLE.schema(), standard=None).standard is None

    @pytest.mark.parametrize("template", _VARIANTS, ids=lambda t: t.schema_name({}))
    def test_every_ops_standard_is_saved_as_standalone_modules(
        self, template: Schema, tmp_path: Path
    ) -> None:
        template.standard.save(tmp_path)
        for path in tmp_path.glob("*.py"):
            assert callable(runpy.run_path(str(path))["run"]), path.name


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
        case = OpSample(
            fused_moe,
            "fused_moe_gate_up_interleaved",
            {
                "n_tokens": 4,
                "model_dim": 512,
                "n_experts": 8,
                "topk": 2,
                "inter_dim": 256,
                "global_experts": None,
            },
        )
        schema = case.template.for_call(case.call())
        assert {n: schema.inputs[n].dtype for n in ("w1", "w2", "w1_scale")} == {
            "w1": "float4_e2m1fn_x2",
            "w2": "float4_e2m1fn_x2",
            "w1_scale": "float8_e8m0fnu",
        }

    def test_absent_optional_input_has_no_dtype(self) -> None:
        schema = GEMM_SAMPLE.schema()
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
            pytest.param({"requires": "at least 3.8"}, id="requires_no_specifier"),
        ],
    )
    def test_rejects(self, change: dict) -> None:
        with pytest.raises(ValidationError):
            _kernel(**change)


class TestSchemaCollection:
    def test_add_workloads_keeps_one_per_axes(self, tmp_path: Path) -> None:
        schema = GEMM_SAMPLE.schema()
        SchemaCollection.from_path(str(tmp_path)).add_workloads(
            [(schema, Workload(axes={"M": m})) for m in (4, 4, 8)]
        )
        collection = SchemaCollection.from_path(str(tmp_path))
        assert [s.workloads for s in collection.schemas[schema.name]] == [
            [Workload(axes={"M": 4}), Workload(axes={"M": 8})]
        ]

    def test_same_name_files_must_agree(self, tmp_path: Path) -> None:
        schema = GEMM_SAMPLE.schema()
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
