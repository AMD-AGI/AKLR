# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import aklr.ops
from aklr.core.schema import (
    Build,
    Evaluation,
    Kernel,
    Schema,
    SourceFile,
    Workload,
    save_json_file,
)
from aklr.core.validation import AuditStage, ValidationScope
from aklr.core.validation.command import main
from aklr.ops.gemm.schema import GEMM_A8W8_NT
from tests.op_case import OpCase

_RUN = "def run(a, b, bias, out):\n"
_COPY = "    out.copy_(a @ b.mT)\n"
_LAUNCH = "    gemm<<<"
_HIP = """\
// out = a @ b.T in bf16, a thread an output element.
#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <optional>
#include <torch/library.h>

namespace {

__global__ void gemm(const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *out,
                     int k) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a[blockIdx.x * k + i]) * __bfloat162float(b[threadIdx.x * k + i]);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = __float2bfloat16(acc);
}

void run(const at::Tensor &a, const at::Tensor &b, const std::optional<at::Tensor> &bias,
         at::Tensor &out) {
    gemm<<<a.size(0), b.size(0), 0, at::hip::getCurrentHIPStream()>>>(
        static_cast<const __hip_bfloat16 *>(a.data_ptr()),
        static_cast<const __hip_bfloat16 *>(b.data_ptr()),
        static_cast<__hip_bfloat16 *>(out.data_ptr()), static_cast<int>(a.size(1)));
}

} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, Tensor? bias, Tensor(a!) out) -> ()", &run);
}
"""
_PASSED = Evaluation(hardware="gfx950/MI355X", accuracy="passed")


def _py(prelude: str = "", body: str = _COPY, signature: str = _RUN) -> str:
    return f'"""K."""\n{prelude}\n\n{signature}{body}'


_PLAIN = _py()


def _schema(sources: dict[str, str]) -> Schema:
    entry = next(iter(sources))
    kernel = Kernel(
        name="kernel",
        author="tests",
        build=Build(
            language="hip" if entry.endswith(".hip") else "triton",
            target_hardware=["gfx950"],
            entry_point=f"{entry}::run",
        ),
        sources=[SourceFile(path=p, content=c) for p, c in sources.items()],
    )
    return OpCase.from_id("gemm_bf16_nt").schema(kernel, [Workload(axes={"M": 4})])


@pytest.fixture
def audit(tmp_path: Path) -> Callable[[dict[str, str]], tuple]:
    def findings(sources: dict[str, str]) -> tuple:
        schema = _schema(sources)
        directory = tmp_path / "aklr/ops/gemm/impls" / schema.kernel.build.language
        path = directory / schema.name / f"{schema.uuid}.json"
        save_json_file(schema, path)
        scope = ValidationScope(tmp_path, "", None, (path,), ())
        return AuditStage().run(scope, None).findings

    return findings


class TestAuditStage:
    def test_committed_schemas_pass(self) -> None:
        root = Path(aklr.ops.__file__).parents[2]
        paths = tuple(root.glob("aklr/ops/*/impls/*/*/*.json"))
        scope = ValidationScope(root, "", None, paths, ())
        assert AuditStage().run(scope, None).findings == ()

    @pytest.mark.parametrize(
        ("relative", "detail"),
        [
            pytest.param(
                "aklr/ops/gemm/kernels/0123456789abcdef.json",
                "not at aklr/ops/<op>/impls/<language>/<name>/<uuid>.json",
                id="outside_impls",
            ),
            pytest.param(
                "aklr/ops/gemm/impls/triton/gemm_bf16_nt/0123456789abcdef.json",
                "the directory is not the name gemm_bf16_nt_n32_k64",
                id="directory_not_the_name",
            ),
            pytest.param(
                "aklr/ops/gemm/impls/flydsl/gemm_bf16_nt_n32_k64/0123456789abcdef.json",
                "not under impls/triton",
                id="directory_of_another_language",
            ),
        ],
    )
    def test_rejects_a_schema_off_its_path(self, tmp_path: Path, relative: str, detail: str) -> None:
        path = tmp_path / relative
        save_json_file(_schema({"kernel.py": _PLAIN}), path)
        scope = ValidationScope(tmp_path, "", None, (path,), ())
        findings = AuditStage().run(scope, None).findings
        assert [(f.code, f.detail) for f in findings] == [("schema_invalid", detail)]

    @pytest.mark.parametrize(
        ("make", "detail"),
        [
            pytest.param(
                lambda s: replace(s, name="gemm_bf16_nt_unnamed"),
                "no schema template is named gemm_bf16_nt_unnamed",
                id="unnamed",
            ),
            pytest.param(
                lambda s: replace(GEMM_A8W8_NT, kernel=s.kernel, workloads=s.workloads),
                "a schema with a kernel has no init to draw its inputs",
                id="no_init",
            ),
            pytest.param(
                lambda s: replace(s, inputs={n: t for n, t in s.inputs.items() if n != "bias"}),
                "inputs ['a', 'b'] are not its template's ['a', 'b', 'bias']",
                id="inputs_drifted",
            ),
        ],
    )
    def test_rejects_a_schema_its_template_cannot_bench(
        self, tmp_path: Path, make: Callable[[Schema], Schema], detail: str
    ) -> None:
        schema = make(_schema({"kernel.py": _PLAIN}))
        path = tmp_path / f"aklr/ops/gemm/impls/triton/{schema.name}/{'1':0>16}.json"
        save_json_file(schema, path)
        scope = ValidationScope(tmp_path, "", None, (path,), ())
        findings = AuditStage().run(scope, None).findings
        assert [(f.code, f.detail) for f in findings] == [("schema_invalid", detail)]

    @pytest.mark.parametrize(
        ("kernel_name", "evaluations", "codes"),
        [
            pytest.param("kernel", [_PASSED], {"perf_edited"}, id="measured_committed"),
            pytest.param("another", [], set(), id="new"),
            pytest.param("another", [_PASSED], {"perf_edited"}, id="new_measured"),
        ],
    )
    def test_only_the_bot_writes_evaluations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kernel_name: str,
        evaluations: list[Evaluation],
        codes: set[str],
    ) -> None:
        committed = _schema({"kernel.py": _PLAIN})
        changed = replace(
            committed,
            kernel=replace(committed.kernel, name=kernel_name),
            workloads=[Workload(axes={"M": 4}, evaluations=evaluations)],
        )
        directory = tmp_path / "aklr/ops/gemm/impls/triton/gemm_bf16_nt_n32_k64"
        save_json_file(committed, directory / f"{committed.uuid}.json")
        for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv],
                cwd=tmp_path,
                check=True,
            )
        save_json_file(changed, directory / f"{changed.uuid}.json")
        monkeypatch.chdir(tmp_path)
        scope = ValidationScope.of_working_tree("HEAD", None)
        assert {f.code for f in AuditStage().run(scope, None).findings} == codes

    def test_audits_every_file_the_runtime_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "aklr/ops/gemm/impls/triton/gemm_bf16_nt_n32_k64"
        save_json_file(_schema({"kernel.py": _PLAIN}), directory / "kernel.json")
        for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv],
                cwd=tmp_path,
                check=True,
            )
        monkeypatch.chdir(tmp_path)
        scope = ValidationScope.of_working_tree(None, None)
        assert [f.detail for f in AuditStage().run(scope, None).findings] == [
            "not at aklr/ops/<op>/impls/<language>/<name>/<uuid>.json"
        ]

    @pytest.mark.parametrize(
        "repeated", [{"batch_size": 3, "M": 4}, {"M": 4, "batch_size": 3}], ids=["same_order", "reordered"]
    )
    def test_two_workloads_of_one_key_are_a_schema_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repeated: dict[str, int]
    ) -> None:
        workloads = [Workload(axes={"batch_size": 3, "M": 4}), Workload(axes=repeated)]
        schema = OpCase.from_id("bmm_bf16_nt").schema(_schema({"kernel.py": _PLAIN}).kernel, workloads)
        path = tmp_path / f"aklr/ops/gemm/impls/triton/{schema.name}/{schema.uuid}.json"
        save_json_file(schema, path)
        for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=tmp_path, check=True
            )
        monkeypatch.chdir(tmp_path)
        assert main(["--stages", "audit", "--out", str(tmp_path / "out"), str(path)]) == 1
        (audit,) = json.loads((tmp_path / "out/report.json").read_text())["stages"]
        text = ",".join(f"{a}={v}" for a, v in repeated.items())
        assert [(f["code"], f["detail"]) for f in audit["findings"]] == [
            ("schema_invalid", f"workload {{{text}}}: repeats")
        ]


class TestPythonKernel:
    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(_PLAIN, id="plain"),
            pytest.param(
                _py(
                    "import torch\n",
                    "    _v, _i = torch.sort(a)\n    torch.add(a, 1)\n" + _COPY,
                ),
                id="pure_torch_functions_of_an_input",
            ),
            pytest.param(
                _py(
                    "import torch\n",
                    "    t = a @ b.mT\n    torch.nn.functional.relu_(t)\n    out.copy_(t)\n",
                ),
                id="inplace_function_on_a_local",
            ),
            pytest.param(
                _py(
                    "import torch\n",
                    "    ws = torch.empty_like(out)\n"
                    "    n = torch.zeros(1, device=a.device)\n"
                    "    torch.matmul(a, b.mT, out=ws)\n    out.copy_(ws + n)\n",
                ),
                id="workspace_allocated_in_launch",
            ),
            pytest.param(
                _py(
                    "from collections.abc import Sequence\n\n\n"
                    "def _shape(s: Sequence[int]) -> tuple[int, ...]:\n"
                    "    return tuple(s)\n"
                ),
                id="collections_abc",
            ),
            pytest.param(
                _py(
                    "from flydsl._mlir import ir\n\n\ndef _f(t=ir.Type):\n"
                    "    from flydsl._mlir import ir\n\n    return ir\n"
                ),
                id="flydsl_imported_twice",
            ),
            pytest.param(_py("for _g in (256, 512):\n    _LAST = _g\n"), id="for_over_constants"),
            pytest.param(_py("for _g in range(4):\n    _LAST = _g\n"), id="for_over_range"),
            pytest.param(
                _py('import torch\n\n\ndef _fp8():\n    return getattr(torch, "float8_e4m3fnuz", None)\n'),
                id="getattr_optional_dtype",
            ),
            pytest.param(_py("_X: tuple[int, ...] = (1,)\n"), id="annotated_constant"),
            pytest.param(
                _py("import flydsl.expr as fx\n\n_N = fx.Int32(fx.Int32(4) * 2)\n"),
                id="nested_flydsl_constant",
            ),
            pytest.param(
                _py("match (1, 2):\n    case (_P, _Q):\n        pass\n"),
                id="match_on_constant",
            ),
            pytest.param(
                _py(
                    "import functools\n\n\n@functools.lru_cache(maxsize=None)\n"
                    "def _compile(n):\n    return n\n\n\n_N = 4\n"
                ),
                id="lru_cache_compile_function",
            ),
            pytest.param(
                _py(
                    "import functools\n\n\n@functools.cache\ndef _compile(n):\n"
                    "    return [n]\n\n\nclass _Runner:\n    def __init__(self, n):\n"
                    "        self.exe = _compile(n)\n        self.n = n\n"
                ),
                id="attribute_holds_a_compile_result",
            ),
            pytest.param(
                _py(
                    "import triton.language as tl\nimport triton\n\n\n@triton.jit\n"
                    "def _k(X, Y):\n    tl.store(Y, tl.load(X).item())\n",
                    "    _k[(1,)](a, out)\n",
                ),
                id="item_in_a_triton_jit_body",
            ),
            pytest.param(
                _py(
                    "import triton.language as tl\nimport triton\nimport triton.compiler"
                    "\n\n\n@triton.jit\ndef _k(X, Y):\n"
                    "    tl.store(Y, tl.load(X).item())\n",
                    "    _k[(1,)](a, out)\n",
                ),
                id="item_in_a_triton_jit_body_with_a_submodule_imported",
            ),
            pytest.param(
                _py(
                    "import triton.language as tl\n"
                    "from triton.experimental import gluon\n\n\n@gluon.jit\n"
                    "def _k(X, Y):\n    tl.store(Y, tl.load(X).item())\n",
                    "    _k[(1,)](a, out)\n",
                ),
                id="item_in_a_gluon_jit_body",
            ),
        ],
    )
    def test_passes(self, audit: Callable, source: str) -> None:
        assert audit({"kernel.py": source}) == ()

    @pytest.mark.parametrize(
        ("source", "rule"),
        [
            pytest.param(
                _py("_X = " + "+".join(["1"] * 3000) + "\n"),
                "SA0",
                id="deeply_nested_expression",
            ),
            pytest.param(
                _py("_X = " + "-" * 8000 + "1\n"),
                "SA0",
                id="expression_too_deep_to_parse",
            ),
            pytest.param(
                _py("_CALLS = 0\n", "    global _CALLS\n    _CALLS += 1\n" + _COPY),
                "SA1",
                id="global_statement",
            ),
            pytest.param(_py(body="    a.zero_()\n"), "SA2", id="writes_input"),
            pytest.param(
                _py(body="    a += 7\n" + _COPY),
                "SA2",
                id="augmented_assignment_to_input",
            ),
            pytest.param(
                _py("import torch\n", "    torch.relu_(a)\n" + _COPY),
                "SA2",
                id="writes_input_through_module_function",
            ),
            pytest.param(
                _py("import torch\n", "    torch.mul(b, 0, out=b)\n" + _COPY),
                "SA2",
                id="writes_input_through_out_keyword",
            ),
            pytest.param(
                _py("from torch import relu_\n", "    relu_(a)\n" + _COPY),
                "SA2",
                id="writes_input_through_imported_function",
            ),
            pytest.param(
                _py(
                    "import torch\n",
                    "    torch.nn.functional.relu(a, inplace=True)\n" + _COPY,
                ),
                "SA2",
                id="writes_input_through_inplace_keyword",
            ),
            pytest.param(
                _py("import torch\n", "    torch.ops.aten.fill_.Scalar(a, 9)\n" + _COPY),
                "SA2",
                id="writes_input_through_op_overload",
            ),
            pytest.param(
                _py("import torch\n", "    torch.manual_seed(0)\n" + _COPY),
                "SA3",
                id="reseeds_global_rng",
            ),
            pytest.param(
                _py(
                    "import functools\nimport torch\n\n\n@functools.cache\n"
                    "def _compile(n):\n"
                    '    torch.set_float32_matmul_precision("medium")\n    return n\n',
                    "    _compile(4)\n" + _COPY,
                ),
                "SA3",
                id="sets_global_torch_state_in_compile_function",
            ),
            pytest.param(
                _py(signature="def run(a, b):\n", body="    ...\n"),
                "SA4",
                id="parameters_other_than_the_call_keys",
            ),
            pytest.param(
                _py(signature="def run(*args, **kwargs):\n", body="    ...\n"),
                "SA4",
                id="var_positional_and_var_keyword",
            ),
            pytest.param(
                _py(signature="def run(*, a, b, bias, out):\n"),
                "SA4",
                id="keyword_only",
            ),
            pytest.param(
                _py(signature="def run(a, b, bias, out, **_):\n"),
                "SA4",
                id="extra_var_keyword",
            ),
        ],
    )
    def test_rejects(self, audit: Callable, source: str, rule: str) -> None:
        findings = audit({"kernel.py": source})
        assert {(f.code, f.detail.split()[0]) for f in findings} == {("source_rejected", rule)}


class TestHipKernel:
    @pytest.mark.parametrize(
        "sources",
        [
            pytest.param({"kernel.hip": _HIP}, id="plain"),
            pytest.param(
                {
                    "kernel.hip": _HIP.replace("#include <optional>\n", '#include "common/util.h"\n'),
                    "common/util.h": '#pragma once\n#include "../common/math.h"\n',
                    "common/math.h": "#pragma once\n#include <optional>\n",
                },
                id="includes_its_own_headers",
            ),
        ],
    )
    def test_passes(self, audit: Callable, sources: dict[str, str]) -> None:
        assert audit(sources) == ()

    @pytest.mark.parametrize(
        ("old", "new", "rule"),
        [
            pytest.param("} // namespace\n", "} /* namespace\n", "SH0", id="unterminated_comment"),
            pytest.param(
                "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {",
                "TORCH_LIBRARY(glm5, m) {",
                "SH1",
                id="literal_namespace",
            ),
            pytest.param(
                "#define TORCH_LIBRARY_EXPAND",
                "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, n) {}\n#define TORCH_LIBRARY_EXPAND",
                "SH1",
                id="second_registration",
            ),
            pytest.param(
                "#include <optional>\n",
                "#include <fstream>\n",
                "SH2",
                id="header_not_allowed",
            ),
            pytest.param(
                "#include <optional>\n",
                '#include "missing.h"\n',
                "SH2",
                id="include_not_a_source",
            ),
            pytest.param(
                _LAUNCH,
                "    (void)hipDeviceSynchronize();\n" + _LAUNCH,
                "SH3",
                id="synchronizes",
            ),
            pytest.param(
                _LAUNCH,
                '    (void)getenv("X");\n' + _LAUNCH,
                "SH3",
                id="reads_the_environment",
            ),
            pytest.param(
                "void run(",
                "static int calls = 0;\n\nvoid run(",
                "SH4",
                id="static_state",
            ),
            pytest.param(
                "__global__",
                "__device__ int counter;\n\n__global__",
                "SH4",
                id="device_state",
            ),
            pytest.param(
                "// out = a @ b.T in bf16, a thread an output element.\n",
                "",
                "SH5",
                id="no_leading_comment",
            ),
        ],
    )
    def test_rejects(self, audit: Callable, old: str, new: str, rule: str) -> None:
        findings = audit({"kernel.hip": _HIP.replace(old, new)})
        assert {(f.code, f.detail.split()[0]) for f in findings} == {("source_rejected", rule)}
