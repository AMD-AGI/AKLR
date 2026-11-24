# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A kernel's entry re-cut to other call keys: params renamed, a dropped one bound to its constant."""

import pytest

from codegen.kernel_entry import KernelEntry

_PYTHON = '''\
import torch


def _launch(a, b, out):
    out.copy_(a @ b.mT)


def run(*, a, b, trans_b, bias, out_dtype, out):
    """out = a @ b.T."""
    if not trans_b or bias is not None:
        raise ValueError("only trans_b")
    if a.shape[0] > 4:
        run(a=a[:4], b=b, trans_b=trans_b, bias=bias, out_dtype=out_dtype, out=out[:4])
        return
    _launch(a, b, out=out)
'''

_HIP = """\
void run(const at::Tensor& a, const at::Tensor& b, bool trans_b, std::optional<at::ScalarType> out_dtype,
         at::Tensor& out) {
    TORCH_CHECK(trans_b, "b must be [n, k]");
    TORCH_CHECK(a.is_cuda(), "tensors must share a device");
    launch(a, b, out);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, bool trans_b, ScalarType? out_dtype, Tensor(a!) out) -> ()", &run);
}
"""


def test_a_python_entry_takes_the_new_names_and_binds_what_is_no_longer_passed() -> None:
    entry = KernelEntry("flydsl", "kernel.py::run", {"kernel.py": _PYTHON})
    sources = entry.retargeted(
        {"a": "A", "b": "B", "out_dtype": "otype"}, ("A", "B", "bias", "otype", "out"), {"trans_b": "True"}
    )
    assert sources["kernel.py"] == _PYTHON.replace(
        '''def run(*, a, b, trans_b, bias, out_dtype, out):
    """out = a @ b.T."""
    if not trans_b or bias is not None:
        raise ValueError("only trans_b")
    if a.shape[0] > 4:
        run(a=a[:4], b=b, trans_b=trans_b, bias=bias, out_dtype=out_dtype, out=out[:4])
        return
    _launch(a, b, out=out)
''',
        '''def run(*, A, B, bias, otype, out):
    """out = a @ b.T."""
    trans_b = True
    if not trans_b or bias is not None:
        raise ValueError("only trans_b")
    if A.shape[0] > 4:
        run(A=A[:4], B=B, bias=bias, otype=otype, out=out[:4])
        return
    _launch(A, B, out=out)
''',
    )


def test_a_python_entry_whose_names_are_taken_is_refused() -> None:
    entry = KernelEntry(
        "flydsl", "kernel.py::run", {"kernel.py": "def run(*, a, out):\n    x = a\n    out.copy_(x)\n"}
    )
    with pytest.raises(ValueError, match="already names \\['x'\\]"):
        entry.retargeted({"a": "x"}, ("x", "out"), {})


def test_a_hip_entry_registers_the_new_names_and_keeps_its_alias_sets() -> None:
    entry = KernelEntry("hip", "kernel.hip::run", {"kernel.hip": _HIP})
    sources = entry.retargeted(
        {"a": "A", "b": "B", "out_dtype": "otype"}, ("A", "B", "otype", "out"), {"trans_b": "True"}
    )
    assert (
        sources["kernel.hip"]
        == """\
void run(const at::Tensor& A, const at::Tensor& B, std::optional<at::ScalarType> otype,
         at::Tensor& out) {
    const bool trans_b = true;
    TORCH_CHECK(trans_b, "b must be [n, k]");
    TORCH_CHECK(A.is_cuda(), "tensors must share a device");
    launch(A, B, out);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor A, Tensor B, ScalarType? otype, Tensor(a!) out) -> ()", &run);
}
"""
    )


def test_a_hip_function_whose_names_are_taken_keeps_them_under_the_new_schema() -> None:
    """The op binds the schema's params to the function's by position."""
    hip = _HIP.replace("launch(a, b, out);", "const auto x = a.size(0);\n    launch(a, b, out);")
    entry = KernelEntry("hip", "kernel.hip::run", {"kernel.hip": hip})
    sources = entry.retargeted({"a": "x", "b": "w"}, ("x", "w", "trans_b", "out_dtype", "out"), {})
    assert (
        'm.def("run(Tensor x, Tensor w, bool trans_b, ScalarType? out_dtype, Tensor(a!) out) -> ()"'
        in sources["kernel.hip"]
    )
    assert "void run(const at::Tensor& a, const at::Tensor& b, bool trans_b" in sources["kernel.hip"]
