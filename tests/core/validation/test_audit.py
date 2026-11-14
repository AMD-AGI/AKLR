# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest

from aklr.core.validation import AuditStage, ValidationScope
from aklr.core.validation.stages._audit._hip import check_hip_kernel
from aklr.core.validation.stages._audit._hsaco import check_hsaco_kernel
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.core.validation.stages._report._finding import Finding
from tests.op_case import HIP_GEMM_KERNEL_SOURCE, HSACO_GEMM_KERNEL_SOURCE, OpCase

_SIGNATURE = "def run(*, a, b, trans_b, bias, out_dtype, out):"  # gemm@1.0's call keys
_PLAIN_SOURCE = '"""K."""\n' + _SIGNATURE + "\n    out.copy_(a @ b.mT)\n"


def _findings(source: str) -> tuple[Finding, ...]:
    """``check_kernel`` on a gemm@1.0 schema whose ``kernel.py`` is ``source``."""
    case = OpCase.of("gemm@1.0")
    schema, contract = case.schema(source), case.contract
    return check_kernel(
        str(schema.relative_path()),
        schema.kernel,
        call_keys=schema.kernel_call_keys,
        tensor_names=tuple(contract.tensors()),
        mutable_names=tuple(contract.mutable_tensors),
    )


class TestCheckKernel:
    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(_PLAIN_SOURCE, id="plain"),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    _v, _i = torch.sort(a)\n    torch.add(a, 1)\n"
                + "    out.copy_(a @ b.mT)\n",
                id="pure_torch_functions_of_an_input",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    t = a @ b.mT\n    torch.nn.functional.relu_(t)\n"
                + "    out.copy_(t)\n",
                id="inplace_function_on_a_local",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    ws = torch.empty_like(out)\n    n = torch.zeros(1, device=a.device)\n"
                + "    torch.matmul(a, b.mT, out=ws)\n    out.copy_(ws + n)\n",
                id="workspace_allocated_in_launch",
            ),
        ],
    )
    def test_plain_kernel_passes(self, source: str) -> None:
        """A plain gemm has no finding."""
        assert _findings(source) == ()

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(
                "from collections.abc import Sequence\n\n\n"
                + "def _shape(s: Sequence[int]) -> tuple[int, ...]:\n    return tuple(s)\n",
                id="collections_abc",
            ),
            pytest.param(
                "from flydsl._mlir import ir\n\n\n"
                + "def _f(t=ir.Type):\n    from flydsl._mlir import ir\n\n    return ir\n",
                id="flydsl_imported_twice",
            ),
            pytest.param("for _g in (256, 512):\n    _LAST = _g\n", id="for_over_constants"),
            pytest.param("for _g in range(4):\n    _LAST = _g\n", id="for_over_range"),
            pytest.param(
                "import torch\n\n\ndef _fp8():\n" + '    return getattr(torch, "float8_e4m3fnuz", None)\n',
                id="getattr_optional_dtype",
            ),
            pytest.param("_X: tuple[int, ...] = (1,)\n", id="annotated_constant"),
            pytest.param(
                "import flydsl.expr as fx\n\n_N = fx.Int32(fx.Int32(4) * 2)\n",
                id="nested_flydsl_constant",
            ),
            pytest.param(
                "match (1, 2):\n    case (_P, _Q):\n        pass\n",
                id="match_on_constant",
            ),
            pytest.param(
                "import functools\n\n\n@functools.lru_cache(maxsize=None)\n"
                + "def _compile(n):\n    return n\n\n\n_N = 4\n",
                id="lru_cache_compile_function",
            ),
            pytest.param(
                "import functools\n\n\n@functools.cache\ndef _compile(n):\n"
                + "    return [n]\n\n\nclass _Runner:\n    def __init__(self, n):\n"
                + "        self.exe = _compile(n)\n        self.n = n\n",
                id="attribute_holds_a_compile_result",
            ),
        ],
    )
    def test_allowed_source_passes(self, source: str) -> None:
        """Sources the README allows have no finding."""
        assert _findings('"""K."""\n' + source + "\n\n" + _SIGNATURE + "\n    out.copy_(a @ b.mT)\n") == ()

    @pytest.mark.parametrize(
        "header",
        [
            pytest.param("import triton\n\n@triton.jit", id="triton_jit"),
            pytest.param(
                "import triton\nimport triton.compiler\n\n@triton.jit",
                id="triton_with_submodule",
            ),
            pytest.param("from triton.experimental import gluon\n\n@gluon.jit", id="gluon_jit"),
        ],
    )
    def test_device_body_is_not_launch_code(self, header: str) -> None:
        """An ``.item()`` inside a jitted device function is not an SA3 sync in launch."""
        source = (
            '"""K."""\nimport triton.language as tl\n'
            + header
            + "\ndef _k(X, Y):\n    tl.store(Y, tl.load(X).item())\n\n"
            + _SIGNATURE
            + "\n    _k[(1,)](a, out)\n"
        )
        assert _findings(source) == ()

    def test_triton_jit_body_is_device_code(self) -> None:
        """A ``.to`` inside ``@triton.jit`` converts a tl value; it is no host copy."""
        source = (
            '"""K."""\nimport triton\nimport triton.language as tl\n\n\n'
            "@triton.jit\ndef _copy(a_ptr, out_ptr):\n"
            "    tl.store(out_ptr, tl.load(a_ptr).to(tl.float32))\n\n\n"
            + _SIGNATURE
            + "\n    _copy[(1,)](a, out)\n"
        )
        case = OpCase.of("gemm@1.0")
        schema = case.schema(source)
        findings = check_kernel(
            str(schema.relative_path()),
            schema.kernel,
            call_keys=schema.kernel_call_keys,
            tensor_names=tuple(case.contract.tensors()),
            mutable_names=tuple(case.contract.mutable_tensors),
        )
        assert findings == ()

    @pytest.mark.parametrize(
        ("source", "rule"),
        [
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    torch.relu_(a)\n    out.copy_(a @ b.mT)\n",
                "SA2",
                id="writes_input_through_module_function",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    torch.mul(b, 0, out=b)\n    out.copy_(a @ b.mT)\n",
                "SA2",
                id="writes_input_through_out_keyword",
            ),
            pytest.param(
                '"""K."""\nfrom torch import relu_\n\n\n'
                + _SIGNATURE
                + "\n    relu_(a)\n    out.copy_(a @ b.mT)\n",
                "SA2",
                id="writes_input_through_imported_function",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    torch.nn.functional.relu(a, inplace=True)\n"
                + "    out.copy_(a @ b.mT)\n",
                "SA2",
                id="writes_input_through_inplace_keyword",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    torch.ops.aten.fill_.Scalar(a, 9)\n    out.copy_(a @ b.mT)\n",
                "SA2",
                id="writes_input_through_op_overload",
            ),
            pytest.param(
                '"""K."""\nimport torch\n\n\n'
                + _SIGNATURE
                + "\n    torch.manual_seed(0)\n    out.copy_(a @ b.mT)\n",
                "SA3",
                id="reseeds_global_rng",
            ),
            pytest.param(
                '"""K."""\n' + _SIGNATURE + "\n    a += 7\n    out.copy_(a @ b.mT)\n",
                "SA2",
                id="augmented_assignment_to_input",
            ),
            pytest.param(
                '"""K."""\nimport functools\nimport torch\n\n\n'
                + "@functools.cache\ndef _compile(n):\n"
                + '    torch.set_float32_matmul_precision("medium")\n    return n\n\n\n'
                + _SIGNATURE
                + "\n    _compile(4)\n    out.copy_(a @ b.mT)\n",
                "SA3",
                id="sets_global_torch_state_in_compile_function",
            ),
            pytest.param(
                '"""K."""\n_X = '
                + "+".join(["1"] * 3000)
                + "\n\n\n"
                + _SIGNATURE
                + "\n    out.copy_(a @ b.mT)\n",
                "SA0",
                id="deeply_nested_expression",
            ),
            pytest.param(
                '"""K."""\n_X = ' + "-" * 8000 + "1\n\n\n" + _SIGNATURE + "\n    out.copy_(a @ b.mT)\n",
                "SA0",
                id="expression_too_deep_to_parse",
            ),
            pytest.param(
                '"""K."""\n' + _SIGNATURE + "\n    a.zero_()\n",
                "SA2",
                id="writes_input",
            ),
            pytest.param('"""K."""\ndef run(a, b):\n    ...\n', "SA4", id="bad_entry"),
            pytest.param(
                '"""K."""\ndef run(*args, **kwargs):\n    ...\n',
                "SA4",
                id="var_keyword",
            ),
            pytest.param(
                '"""K."""\n' + _SIGNATURE.replace("*, ", "") + "\n    ...\n",
                "SA4",
                id="not_keyword_only",
            ),
            pytest.param(
                '"""K."""\n' + _SIGNATURE.replace("out):", "out, **_):") + "\n    ...\n",
                "SA4",
                id="extra_kwargs",
            ),
        ],
    )
    def test_rejects_rule_violation(self, source: str, rule: str) -> None:
        """Every finding names ``kernel.py``, and their rules are exactly ``{rule}``."""
        findings = _findings(source)
        assert all(" kernel.py:" in f.detail for f in findings)
        assert {f.detail.split()[0] for f in findings} == {rule}


def _hip_findings(source: str, other_sources: dict[str, str] | None = None) -> tuple[Finding, ...]:
    """``check_hip_kernel`` on a gemm@1.0 hip schema."""
    schema = OpCase.of("gemm@1.0").schema(source, language="hip", other_sources=other_sources)
    return check_hip_kernel(str(schema.relative_path()), schema.kernel)


_LAUNCH = "    gemm<<<"


class TestCheckHipKernel:
    def test_plain_kernel_passes(self) -> None:
        """The hip gemm has no finding, nor with a header that includes a sibling."""
        assert _hip_findings(HIP_GEMM_KERNEL_SOURCE) == ()
        source = HIP_GEMM_KERNEL_SOURCE.replace("#include <optional>\n", '#include "common/util.h"\n')
        headers = {
            "common/util.h": '#pragma once\n#include "../common/math.h"\n',
            "common/math.h": "#pragma once\n#include <optional>\n",
        }
        assert _hip_findings(source, headers) == ()

    @pytest.mark.parametrize(
        ("old", "new", "rule"),
        [
            ("} // namespace\n", "} /* namespace\n", "SH0"),
            (
                "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {",
                "TORCH_LIBRARY(glm5, m) {",
                "SH1",
            ),
            (
                "#define TORCH_LIBRARY_EXPAND",
                "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, n) {}\n#define TORCH_LIBRARY_EXPAND",
                "SH1",
            ),
            ("#include <optional>\n", "#include <fstream>\n", "SH2"),
            ("#include <optional>\n", '#include "missing.h"\n', "SH2"),
            (_LAUNCH, "    (void)hipDeviceSynchronize();\n" + _LAUNCH, "SH3"),
            (_LAUNCH, '    (void)getenv("X");\n' + _LAUNCH, "SH3"),
            ("void run(", "static int calls = 0;\n\nvoid run(", "SH4"),
            ("__global__", "__device__ int counter;\n\n__global__", "SH4"),
            ("// out = a @ b.T in bf16, a thread an output element.\n", "", "SH5"),
        ],
        ids=[
            "unterminated_comment",
            "literal_namespace",
            "second_registration",
            "header_not_allowed",
            "include_not_a_source",
            "synchronizes",
            "reads_the_environment",
            "static_state",
            "device_state",
            "no_leading_comment",
        ],
    )
    def test_rejects_rule_violation(self, old: str, new: str, rule: str) -> None:
        """Every finding names ``kernel.hip``, and their rules are exactly ``{rule}``."""
        source = HIP_GEMM_KERNEL_SOURCE.replace(old, new)
        assert source != HIP_GEMM_KERNEL_SOURCE
        findings = _hip_findings(source)
        assert findings
        assert all(" kernel.hip:" in f.detail for f in findings)
        assert {f.detail.split()[0] for f in findings} == {rule}


_CODE_OBJECTS = {"codeobjects.py": '"""The gemm."""\n\nGEMM = b""\n'}


def _hsaco_findings(source: str) -> tuple[Finding, ...]:
    """``check_kernel`` and ``check_hsaco_kernel`` on a gemm@1.0 hsaco schema."""
    case = OpCase.of("gemm@1.0")
    schema, contract = case.schema(source, language="hsaco", other_sources=_CODE_OBJECTS), case.contract
    schema_path = str(schema.relative_path())
    return check_kernel(
        schema_path,
        schema.kernel,
        call_keys=schema.kernel_call_keys,
        tensor_names=tuple(contract.tensors()),
        mutable_names=tuple(contract.mutable_tensors),
    ) + check_hsaco_kernel(schema_path, schema.kernel)


_HSACO_LAUNCH = "    kernarg = "


class TestCheckHsacoKernel:
    def test_plain_kernel_passes(self) -> None:
        """The hsaco gemm has no finding."""
        assert _hsaco_findings(HSACO_GEMM_KERNEL_SOURCE) == ()

    @pytest.mark.parametrize(
        ("old", "new", "rule"),
        [
            ("import struct\n", "import ctypes\nimport struct\n", "SC1"),
            ("import struct\n", "import os.path\nimport struct\n", "SC1"),
            ("import hsaco\n", "import hip_extension as hsaco\n", "SC1"),
            ("import struct\n", "import struct\nfrom torch.utils import cpp_extension\n", "SC1"),
            (_HSACO_LAUNCH, '    __import__("ctypes")\n' + _HSACO_LAUNCH, "SC2"),
            (_HSACO_LAUNCH, '    eval("a")\n' + _HSACO_LAUNCH, "SC2"),
            (
                "import struct\n",
                'import struct\n\nimport torch\n\ntorch.ops.load_library("gemm.so")\n',
                "SC2",
            ),
            ("import struct\n", "import struct\n\nimport torch as t\n\nt._C._cuda_init()\n", "SC2"),
            ("import struct\n", "import struct\n\nfrom torch import cuda\n\ncuda.cudart()\n", "SC2"),
            ("import struct\n", "import struct\n\nimport torch\n", None),
            (_HSACO_LAUNCH, "    torch.cuda.synchronize()\n" + _HSACO_LAUNCH, "SA3"),
        ],
        ids=[
            "ctypes",
            "os",
            "another_aklr_module",
            "torch_cpp_extension",
            "dunder_import",
            "eval",
            "torch_load_library",
            "torch_c_aliased",
            "torch_cudart_from_import",
            "torch_passes",
            "the_sa_rules_too",
        ],
    )
    def test_rejects_rule_violation(self, old: str, new: str, rule: str | None) -> None:
        """Every finding names ``kernel.py``, and their rules are exactly ``{rule}`` (none for None)."""
        source = HSACO_GEMM_KERNEL_SOURCE.replace(old, new, 1)
        assert source != HSACO_GEMM_KERNEL_SOURCE
        findings = _hsaco_findings(source)
        assert all(" kernel.py:" in f.detail for f in findings)
        assert {f.detail.split()[0] for f in findings} == ({rule} if rule else set())


_BAD_SOURCE = '"""K."""\n' + _SIGNATURE + "\n    a.zero_()\n"
_OLD_FORMAT_PATH = "aklr/ops/gemm/kernels/gemm.py"  # a base that predates the format
# origin/main's solution file layout: under impls/, but its stem is not a schema uuid
_LEGACY_SOLUTION_PATH = "aklr/ops/gemm/impls/triton/fp8/fp8_gfx950_m1.json"


class TestAuditStage:
    """``AuditStage.run`` against a base whose files are given, no git."""

    @pytest.mark.parametrize(
        ("base", "head_source", "head_ms", "measured", "codes"),
        [
            pytest.param(
                (_BAD_SOURCE, [4]), _BAD_SOURCE, [4], False,
                {"source_rejected"},
                id="incumbent_checked",
            ),
            pytest.param(
                (_PLAIN_SOURCE, [4]), _BAD_SOURCE, [4], False,
                {"source_rejected"},
                id="edited_kernel_checked",
            ),
            pytest.param(
                (_BAD_SOURCE, [4]), _BAD_SOURCE, [4, 8], False,
                {"source_rejected"},
                id="incumbent_new_points_checked",
            ),
            pytest.param(
                _OLD_FORMAT_PATH, _BAD_SOURCE, [4], False, {"source_rejected"},
                id="base_without_schema_data_checked",
            ),
            pytest.param(
                _LEGACY_SOLUTION_PATH, _BAD_SOURCE, [4], False,
                {"source_rejected"},
                id="legacy_solution_checked",
            ),
            pytest.param(
                _OLD_FORMAT_PATH, _PLAIN_SOURCE, [4], True, {"perf_edited"},
                id="predates_format_keeps_perf_edited",
            ),
            pytest.param(
                None, _BAD_SOURCE, [4], False, {"source_rejected"},
                id="no_base",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.parametrize("head_author", ["", "TEST_author"])
    def test_base(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        base: tuple[str, list[int]] | str | None,
        head_source: str,
        head_ms: list[int],
        measured: bool,
        codes: set[str],
        head_author: str,
    ) -> None:
        """The codes of the findings of a run whose change set is the head schema and one code path.
        An ``author`` on the head alone changes none: a shipped kernel stays exempt."""
        case = OpCase.of("gemm@1.0")
        op_type = case.contract.op_type

        def schema_at(source: str, ms: list[int], author: str = "") -> tuple[str, str]:
            measurement = {"accuracy": "passed", "speedup": 1.5} if measured else {}
            schema = case.contract.narrow(
                case.point,
                workloads=[{"axes": {"m": m}, **measurement} for m in ms],
                kernel=case.schema(source).kernel.model_copy(update={"author": author}),
                models=["test-model"],
            )
            return f"aklr/ops/{op_type}/{schema.relative_path()}", schema.schema_text()

        head_path, head_text = schema_at(head_source, head_ms, head_author)
        (tmp_path / head_path).parent.mkdir(parents=True)
        (tmp_path / head_path).write_text(head_text, encoding="utf-8")
        base_files = {base: ""} if isinstance(base, str) else dict([schema_at(*base)]) if base else {}
        monkeypatch.setattr(
            ValidationScope,
            "paths_at_base",
            lambda _, directory: tuple(sorted(p for p in base_files if p.startswith(directory + "/"))),
        )
        monkeypatch.setattr(ValidationScope, "text_at_base", lambda _, path: base_files[path])
        monkeypatch.setattr(
            ValidationScope,
            "changed_paths",
            lambda _: tuple(sorted({head_path, "aklr/core/x.py"} | base_files.keys())),
        )
        scope = ValidationScope(
            tmp_path,
            "",
            None if base is None else "base",
            ((tmp_path / head_path).resolve(),),
            (),
        )
        report = AuditStage().run(scope, None)
        assert {f.code.value for f in report.findings} == codes

    def test_hip_kernel_takes_the_sh_rules(self, tmp_path: Path) -> None:
        """A hip schema's kernel is audited by SH0-SH5, not SA0-SA4."""
        case = OpCase.of("gemm@1.0")
        source = HIP_GEMM_KERNEL_SOURCE.replace(_LAUNCH, '    (void)getenv("X");\n' + _LAUNCH)
        schema = case.schema(source, language="hip")
        path = tmp_path / "aklr/ops" / case.contract.op_type / schema.relative_path()
        path.parent.mkdir(parents=True)
        path.write_text(schema.schema_text(), encoding="utf-8")
        scope = ValidationScope(tmp_path, "", None, (path.resolve(),), ())
        (finding,) = AuditStage().run(scope, None).findings
        assert finding.code.value == "source_rejected"
        assert finding.detail.startswith("SH3 kernel.hip:")

    def test_hsaco_kernel_takes_the_sa_and_sc_rules(self, tmp_path: Path) -> None:
        """An hsaco schema's kernel is audited by SA0-SA4, then SC1-SC2."""
        case = OpCase.of("gemm@1.0")
        source = HSACO_GEMM_KERNEL_SOURCE.replace(
            "import struct\n", "import ctypes\nimport struct\n"
        ).replace(_HSACO_LAUNCH, "    a.zero_()\n" + _HSACO_LAUNCH)
        schema = case.schema(source, language="hsaco", other_sources=_CODE_OBJECTS)
        path = tmp_path / "aklr/ops" / case.contract.op_type / schema.relative_path()
        path.parent.mkdir(parents=True)
        path.write_text(schema.schema_text(), encoding="utf-8")
        scope = ValidationScope(tmp_path, "", None, (path.resolve(),), ())
        findings = AuditStage().run(scope, None).findings
        assert {f.code.value for f in findings} == {"source_rejected"}
        assert sorted(f.detail.split(" kernel.py:")[0] for f in findings) == ["SA2", "SC1"]
