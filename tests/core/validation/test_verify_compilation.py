# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from aklr.core.validation.validators.verify import Compilation
from tests.op_case import OpCase

TRITON_GEMM = """
import triton
import triton.language as tl

@triton.jit
def gemm(A, B, Out, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, BK)
    a = tl.load(A + m * K + k, k < K, 0).to(tl.float32)
    b = tl.load(B + n * K + k, k < K, 0).to(tl.float32)
    tl.store(Out + m * N + n, tl.sum(a * b, 0))

def run(*, a, b, trans_b, bias, out_dtype, out):
    gemm[lambda meta: (a.shape[0], b.shape[0])](
        a, b, out, b.shape[0], a.shape[1], triton.next_power_of_2(a.shape[1]))
"""


def compile_schema(tmp_path, schema):
    path = tmp_path / "schema.json"
    path.write_text(schema.schema_text())
    return Compilation.of_schema(tmp_path, path.name, schema, tmp_path / "compiled", 0, 1, None)


def test_triton_workload_produces_isa_and_launch_arguments(tmp_path):
    schema = OpCase.of("gemm@1.0").schema(TRITON_GEMM, language="triton")
    (compilation,) = compile_schema(tmp_path, schema)
    assert compilation.failure == ""
    assert Path(compilation.isa_path).is_file()
    assert "s_endpgm" in Path(compilation.isa_path).read_text()
    launch = compilation.launch
    assert launch["grid"] == [schema.workloads[0].axes["m"], schema.axes["n"].lo, 1]
    assert launch["block"] == [256, 1, 1]
    assert [p for p in launch["params"] if "ptr" in p] == [{"ptr": "a"}, {"ptr": "b"}, {"ptr": "out"}]
    assert {a["name"]: a["kind"] for a in launch["arrays"]} == {"a": "in", "b": "in", "out": "out"}


GLUON_GEMM = """
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

@g.jit
def gemm(A, B, Out, N: gl.constexpr, K: gl.constexpr, BK: gl.constexpr):
    m = gl.program_id(0)
    n = gl.program_id(1)
    k = gl.arange(0, BK, layout=gl.constexpr(BlockedLayout([1], [64], [4], [0])))
    a = gl.load(A + m * K + k, k < K, 0).to(gl.float32)
    b = gl.load(B + n * K + k, k < K, 0).to(gl.float32)
    gl.store(Out + m * N + n, gl.sum(a * b, 0))

def run(*, a, b, trans_b, bias, out_dtype, out):
    gemm[(a.shape[0], b.shape[0])](
        a, b, out, b.shape[0], a.shape[1], triton.next_power_of_2(a.shape[1]))
"""


def test_gluon_workload_produces_isa_and_launch_arguments(tmp_path):
    schema = OpCase.of("gemm@1.0").schema(GLUON_GEMM, language="triton")
    (compilation,) = compile_schema(tmp_path, schema)
    assert compilation.failure == ""
    assert "s_endpgm" in Path(compilation.isa_path).read_text()
    assert compilation.launch["block"] == [256, 1, 1]
    assert [p for p in compilation.launch["params"] if "ptr" in p] == [
        {"ptr": "a"},
        {"ptr": "b"},
        {"ptr": "out"},
    ]


def test_multiple_triton_launches_preserve_workspace_identity(tmp_path):
    source = TRITON_GEMM.replace("import triton\n", "import triton\nimport torch\n", 1)
    source = source.replace(
        "def run(*,",
        """@triton.jit
def copy(A, Tmp, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    tl.store(Tmp + row * K + k, tl.load(A + row * K + k, k < K, 0), k < K)

def run(*,""",
    )
    source = source.replace(
        "    gemm[lambda",
        """    workspace = torch.empty_like(a)
    copy[(a.shape[0],)](a, workspace, a.shape[1], triton.next_power_of_2(a.shape[1]))
    a = workspace
    gemm[lambda""",
    )
    schema = OpCase.of("gemm@1.0").schema(source, language="triton")
    first, second = compile_schema(tmp_path, schema)
    assert first.failure == second.failure == ""
    workspace = next(a for a in first.launch["arrays"] if a["name"] != "a")
    assert workspace["kind"] == "out"
    assert next(a for a in second.launch["arrays"] if a["name"] == workspace["name"])["kind"] == "in"
    assert {"ptr": workspace["name"]} in second.launch["params"]


def test_triton_launch_packs_runtime_scalars_at_isa_offsets(tmp_path):
    source = TRITON_GEMM.replace("@triton.jit", '@triton.jit(do_not_specialize=["length", "gain"])')
    source = source.replace(
        "N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr",
        "length, gain, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr",
    )
    source = source.replace("tl.sum(a * b, 0)", "tl.sum(a * b, 0) * gain + length")
    source = source.replace("a, b, out, b.shape[0]", "a, b, out, 7, 0.25, b.shape[0]")
    schema = OpCase.of("gemm@1.0").schema(source, language="triton")
    (compilation,) = compile_schema(tmp_path, schema)
    assert compilation.failure == ""
    assert compilation.launch["params"][:5] == [
        {"ptr": "a"},
        {"ptr": "b"},
        {"ptr": "out"},
        {"int": 7},
        {"int": 1048576000},
    ]
