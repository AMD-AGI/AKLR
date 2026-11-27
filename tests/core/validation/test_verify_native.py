# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aklr.core.validation.validators.verify import Compilation
from tests.op_case import HSACO_GEMM_DEVICE_SOURCE, HSACO_GEMM_KERNEL_SOURCE, OpCase


@pytest.fixture
def rocm_tools():
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    tools = {}
    for name, directory in (("hipcc", "bin"), ("llvm-mc", "llvm/bin")):
        installed = root / directory / name
        path = str(installed) if installed.is_file() else shutil.which(name)
        if path is None:
            pytest.skip(f"offline native compilation requires {name}")
        tools[name] = path
    return tools


@pytest.fixture
def native_schema(tmp_path, request, rocm_tools):
    arch = getattr(request, "param", "gfx950")
    source = tmp_path / "gemm.hip"
    binary = tmp_path / "gemm.hsaco"
    source.write_text(HSACO_GEMM_DEVICE_SOURCE)
    subprocess.run(
        [rocm_tools["hipcc"], "--genco", f"--offload-arch={arch}", "-O3", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    encoded = base64.b64encode(binary.read_bytes()).decode()
    return OpCase.of("gemm@1.0").schema(
        HSACO_GEMM_KERNEL_SOURCE,
        language="hsaco",
        other_sources={"codeobjects.py": f'import base64\nGEMM = base64.b64decode("{encoded}")\n'},
    )


def test_hsaco_records_binary_isa_and_launch_without_a_gpu(tmp_path, native_schema, rocm_tools):
    path = tmp_path / "schema.json"
    path.write_text(native_schema.schema_text())
    (compiled,) = Compilation.of_schema(tmp_path, path.name, native_schema, tmp_path / "compiled", 0, 1, None)
    assert compiled.failure == ""
    isa = Path(compiled.isa_path).read_text()
    assert "s_endpgm" in isa
    assert ".amdhsa_kernel gemm" in isa
    assert ".amdgpu_metadata" in isa
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"' in isa
    subprocess.run(
        [
            rocm_tools["llvm-mc"],
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "-filetype=obj",
            compiled.isa_path,
            "-o",
            str(tmp_path / "reconstructed.o"),
        ],
        check=True,
        capture_output=True,
    )
    assert compiled.kernel == "gemm"
    assert compiled.launch["grid"] == [native_schema.workloads[0].axes["m"], 1, 1]
    assert compiled.launch["block"] == [native_schema.axes["n"].lo, 1, 1]
    assert [p for p in compiled.launch["params"] if "ptr" in p] == [
        {"ptr": "a"},
        {"ptr": "b"},
        {"ptr": "out"},
    ]
    assert {a["name"]: a["kind"] for a in compiled.launch["arrays"]} == {"a": "in", "b": "in", "out": "out"}


@pytest.mark.parametrize(
    ("replacement", "failure"),
    [
        ("kernarg = kernarg[:8]", "kernarg buffer size"),
        (
            'kernarg = struct.pack("<QQQii", 123, b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])',
            "unknown pointer",
        ),
    ],
    ids=["truncated-arguments", "unowned-pointer"],
)
def test_hsaco_unsupported_arguments_are_failures(tmp_path, native_schema, replacement, failure):
    source = HSACO_GEMM_KERNEL_SOURCE.replace("    _GEMM.launch", f"    {replacement}\n    _GEMM.launch")
    schema = OpCase.of("gemm@1.0").schema(
        source,
        language="hsaco",
        other_sources={s.path: s.content for s in native_schema.kernel.sources if s.path != "kernel.py"},
    )
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    (compiled,) = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert failure in compiled.failure
    assert compiled.launch is None


@pytest.mark.gfx950
def test_hip_records_launches_and_suppresses_only_extension_kernels(tmp_path):
    from tests.op_case import HIP_GEMM_KERNEL_SOURCE

    source = HIP_GEMM_KERNEL_SOURCE.replace("    gemm<<<", "    out.fill_(7);\n    gemm<<<", 1).replace(
        "static_cast<int>(a.size(1)));",
        """static_cast<int>(a.size(1)));
    gemm<<<a.size(0), b.size(0), 128, at::hip::getCurrentHIPStream()>>>(
        static_cast<const __hip_bfloat16 *>(a.data_ptr()),
        static_cast<const __hip_bfloat16 *>(b.data_ptr()),
        static_cast<__hip_bfloat16 *>(out.data_ptr()), static_cast<int>(a.size(1)));
    TORCH_CHECK(out.eq(7).all().item<bool>(), "verify executed a target kernel");""",
    )
    schema = OpCase.of("gemm@1.0").schema(source, language="hip")
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    first, second = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert first.failure == second.failure == ""
    assert "gemm" in first.kernel
    isa = Path(first.isa_path).read_text()
    assert "s_endpgm" in isa
    assert ".amdhsa_kernel" in isa
    assert ".amdgpu_metadata" in isa
    assert first.launch["grid"] == [schema.workloads[0].axes["m"], 1, 1]
    assert first.launch["block"] == [schema.axes["n"].lo, 1, 1]
    assert first.launch["dynamic_shared_bytes"] == 0
    assert second.launch["dynamic_shared_bytes"] == 128
    assert [p for p in first.launch["params"] if "ptr" in p] == [{"ptr": "a"}, {"ptr": "b"}, {"ptr": "out"}]
    assert {a["name"]: a["kind"] for a in first.launch["arrays"]} == {"a": "in", "b": "in", "out": "out"}


@pytest.mark.parametrize("native_schema", ["gfx942"], indirect=True)
def test_hsaco_rejects_a_code_object_for_another_architecture(tmp_path, native_schema):
    path = tmp_path / "schema.json"
    path.write_text(native_schema.schema_text())
    (compiled,) = Compilation.of_schema(tmp_path, path.name, native_schema, tmp_path / "compiled", 0, 1, None)
    assert "0 code objects for gfx950" in compiled.failure
    assert compiled.launch is None


def test_hsaco_preserves_launch_abi_and_scalar_bytes(tmp_path, native_schema):
    source = HSACO_GEMM_KERNEL_SOURCE.replace("b.shape[0], a.shape[1])", "-7, 25)").replace(
        "_GEMM.launch((a.shape[0],), (b.shape[0],), kernarg, device=a.device)",
        "_GEMM.launch((2, 3, 4), (8, 4, 2), bytearray(kernarg), shared_memory=128, device=a.device)",
    )
    schema = OpCase.of("gemm@1.0").schema(
        source,
        language="hsaco",
        other_sources={s.path: s.content for s in native_schema.kernel.sources if s.path != "kernel.py"},
    )
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    (compiled,) = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert compiled.failure == ""
    assert compiled.launch["grid"] == [2, 3, 4]
    assert compiled.launch["block"] == [8, 4, 2]
    assert compiled.launch["dynamic_shared_bytes"] == 128
    assert compiled.launch["params"][3:5] == [{"int": -7}, {"int": 25}]


@pytest.mark.parametrize(
    "launch",
    ["((0,), (16,), kernarg)", "((1,), (16,), kernarg, shared_memory=-1)"],
    ids=["empty-grid", "negative-shared-memory"],
)
def test_hsaco_rejects_invalid_launch_dimensions(tmp_path, native_schema, launch):
    source = HSACO_GEMM_KERNEL_SOURCE.replace(
        "((a.shape[0],), (b.shape[0],), kernarg, device=a.device)",
        launch,
    )
    schema = OpCase.of("gemm@1.0").schema(
        source,
        language="hsaco",
        other_sources={s.path: s.content for s in native_schema.kernel.sources if s.path != "kernel.py"},
    )
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    (compiled,) = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert "invalid native launch" in compiled.failure
    assert compiled.launch is None


def test_missing_hip_capture_toolchain_is_a_compilation_failure(tmp_path, monkeypatch):
    from tests.op_case import HIP_GEMM_KERNEL_SOURCE

    monkeypatch.setenv("PATH", str(tmp_path))
    schema = OpCase.of("gemm@1.0").schema(HIP_GEMM_KERNEL_SOURCE, language="hip")
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    (compilation,) = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert "C++ compiler is required" in compilation.failure
    assert compilation.launch is None


def test_hsaco_isa_is_accepted_by_volta(tmp_path, native_schema):
    from aklr.core.validation import ValidationScope, VerifyValidator
    from aklr.core.validation.report import PlatformRecord

    if shutil.which("volta") is None:
        pytest.skip("requires the Volta verifier binary")
    path = tmp_path / "schema.json"
    path.write_text(native_schema.schema_text())
    scope = ValidationScope(tmp_path, "1" * 40, None, (path,), (path,))
    report = VerifyValidator(n_workers=1).run(
        scope, PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.5")
    )
    assert report.status == "passed"
    (record,) = report.path_records
    assert {v.status for v in record.verdicts} == {"proved"}


@pytest.mark.gfx950
@pytest.mark.parametrize("language", ["hip", "hsaco"])
def test_compiled_native_kernel_remains_correct_at_runtime(
    tmp_path, builder_registry, native_schema, language
):
    import torch
    from tests.op_case import HIP_GEMM_KERNEL_SOURCE

    case = OpCase.of("gemm@1.0")
    schema = case.schema(HIP_GEMM_KERNEL_SOURCE, language="hip") if language == "hip" else native_schema
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    (compilation,) = Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)
    assert compilation.failure == ""
    inputs = case.inputs()
    call = case.call(inputs, "cuda")
    call["out"] = torch.empty(
        case.contract.storage_shape_of("out", case.point), dtype=call["out_dtype"], device="cuda"
    )
    builder_registry.build(schema)(**call)
    assert case.compare({"out": call["out"]}, inputs) == {}
