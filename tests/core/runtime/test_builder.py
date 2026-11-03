# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.builder import (
    KERNEL_PACKAGE_PREFIX,
    BuilderRegistry,
    BuildError,
    Runnable,
)
from aklr.core.runtime.builder.builder import _taking_call_keys
from tests.op_case import GEMM_KERNEL_SOURCE, HIP_GEMM_KERNEL_SOURCE, OpCase

_IMPORT_ERROR_KERNEL_SOURCE = "raise ImportError('no kernel')\n"
_UNBOUND = r"kernel\.py::run does not take the call keys"


class TestBuilderRegistry:
    """``build`` imports a schema's kernel as a ``Runnable``, once per content hash."""

    def test_imports_entry_point(self, builder_registry: BuilderRegistry) -> None:
        """The gemm case's schema builds to a ``Runnable`` writing ``a @ b.T`` into ``out``."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(GEMM_KERNEL_SOURCE))
        assert isinstance(runnable, Runnable)
        call = case.call(case.inputs())
        call["out"] = torch.empty(case.contract.storage_shape_of("out", case.point), dtype=call["out_dtype"])
        runnable(**call)
        torch.testing.assert_close(call["out"], call["a"] @ call["b"].mT, rtol=0, atol=0)
        package = runnable.metadata.misc["module_name"].partition(".")[0]
        assert package in sys.modules
        builder_registry.cleanup()
        assert package not in sys.modules

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            (_IMPORT_ERROR_KERNEL_SOURCE, r"cannot import kernel\.py::run"),
            ("def run(a, b):\n    return a @ b.mT\n", _UNBOUND),
            ("def run(*, a, b, out, scale, **_): ...\n", _UNBOUND),
        ],
        ids=["import_error", "without_out", "unknown_required"],
    )
    def test_build_error_leaves_no_module(
        self, builder_registry: BuilderRegistry, source: str, match: str
    ) -> None:
        """A kernel that cannot be imported or bound gives a ``BuildError`` naming its entry point, and leaves no package behind."""
        schema = OpCase.of("gemm@1.0").schema(source)
        with pytest.raises(BuildError, match=match):
            builder_registry.build(schema)
        package = f"{KERNEL_PACKAGE_PREFIX}flydsl_{schema.kernel.content_hash[:16]}"
        assert not [m for m in sys.modules if f"{m}.".startswith(f"{package}.")]

    def test_caches_by_content_hash(self, builder_registry: BuilderRegistry) -> None:
        """Two schemas with the same kernel build one ``Runnable``."""
        gemm_schema = OpCase.of("gemm@1.0").schema()
        bmm_schema = OpCase.of("bmm@1.0").schema()
        assert gemm_schema.name != bmm_schema.name
        assert builder_registry.build(gemm_schema) is builder_registry.build(bmm_schema)


def test_an_optional_attr_at_its_default_hides_its_param_from_the_kernel() -> None:
    """Dispatch passes every api param; a kernel built before an optional attr was added takes
    only the call keys."""
    schema = SimpleNamespace(
        kernel_call_keys=("a", "out"),
        api_params=tuple(SimpleNamespace(name=n) for n in ("a", "has_b", "b", "out")),
    )
    assert _taking_call_keys(lambda **call: call, schema)(a=1, has_b=False, b=None, out=2) == {
        "a": 1,
        "out": 2,
    }
    every = SimpleNamespace(kernel_call_keys=("a",), api_params=(SimpleNamespace(name="a"),))
    fn = lambda **call: call
    assert _taking_call_keys(fn, every) is fn


@pytest.mark.gpu
class TestHipBuilder:
    """``build`` of a hip schema loads its library and returns its entry op."""

    def test_builds_entry_op(self, builder_registry: BuilderRegistry) -> None:
        """The hip gemm writes ``a @ b.T`` into ``out``."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(HIP_GEMM_KERNEL_SOURCE, language="hip"))
        assert runnable.metadata.build_type == "hip"
        call = case.call(case.inputs(), device="cuda")
        call["out"] = torch.empty(
            case.contract.storage_shape_of("out", case.point),
            dtype=call["out_dtype"],
            device="cuda",
        )
        runnable(**call)
        torch.testing.assert_close(
            call["out"].float(),
            call["a"].float() @ call["b"].float().mT,
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [
            ("float acc = 0.0f;", "float acc = ;", "Error building extension"),
            ('m.def("run(', 'm.def("other(', "registers no op run"),
            ("Tensor(a!) out)", "Tensor(a!) output)", "not the call keys"),
            ("Tensor(a!) out)", "Tensor out)", r"must write exactly \['out'\]"),
        ],
        ids=["compile_error", "no_op", "other_names", "output_not_written"],
    )
    def test_build_error(self, builder_registry: BuilderRegistry, old: str, new: str, match: str) -> None:
        """A kernel that does not compile, or whose entry op is not the call's, gives a ``BuildError``."""
        source = HIP_GEMM_KERNEL_SOURCE.replace(old, new)
        assert source != HIP_GEMM_KERNEL_SOURCE
        with pytest.raises(BuildError, match=match):
            builder_registry.build(OpCase.of("gemm@1.0").schema(source, language="hip"))

    def test_library_that_does_not_load_is_the_kernels_fault(self, builder_registry: BuilderRegistry) -> None:
        """An undefined symbol links but fails the load: a ``BuildError`` not caused by an ``OSError``, so bench does not blame the platform."""
        source = HIP_GEMM_KERNEL_SOURCE.replace(
            "namespace {\n", "void aklr_undefined();\n\nnamespace {\n", 1
        ).replace("    gemm<<<", "    aklr_undefined();\n    gemm<<<")
        with pytest.raises(BuildError, match="undefined symbol") as error:
            builder_registry.build(OpCase.of("gemm@1.0").schema(source, language="hip"))
        assert not isinstance(error.value.__cause__, OSError)
