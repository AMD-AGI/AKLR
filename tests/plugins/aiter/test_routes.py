# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's GEMM calls to AKLR ops: what is taken and what goes back to aiter."""

import importlib

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import aiter as aklr_aiter

# the package's names are the functions; the modules hold what the tests replace
gemm_module = importlib.import_module("aklr.plugins.aiter.tuned_gemm")
triton_gemm_module = importlib.import_module("aklr.plugins.aiter.triton_gemm")


@pytest.fixture
def op_calls(monkeypatch):
    """The AKLR op calls, each as its keyword arguments."""
    calls = []

    def op(*args, **kwargs):
        calls.append(kwargs)
        return "aklr"

    monkeypatch.setattr(gemm_module, "gemm", op)
    monkeypatch.setattr(triton_gemm_module, "gemm", op)
    return calls


def mm_call(**changed):
    call = {"bias": None, "otype": None, "scale_a": None, "scale_b": None, "scale_c": None} | changed
    return aklr_aiter.mm(torch.zeros(4, 8), torch.zeros(6, 8), **call)


@pytest.mark.parametrize("scale", ["scale_a", "scale_b", "scale_c"])
def test_a_scaled_gemm_goes_back_to_aiter(op_calls, scale):
    assert mm_call(**{scale: torch.ones(1)}) is FallbackReason.OUT_OF_CONTRACT
    assert op_calls == []


def test_a_gemm_reaches_the_op_with_the_callers_buffer(op_calls):
    out = torch.empty(4, 6)
    assert mm_call(otype=torch.bfloat16, out=out) == "aklr"
    (call,) = op_calls
    assert call["out"] is out
    assert call["out_dtype"] is torch.bfloat16


def triton_call(**changed):
    call = {
        "bias": None,
        "dtype": torch.bfloat16,
        "y": None,
        "activation": None,
        "skip_reduce": False,
    } | changed
    return aklr_aiter.gemm_a16w16(torch.zeros(4, 8), torch.zeros(6, 8), **call)


@pytest.mark.parametrize(
    "changed", [{"activation": "gelu"}, {"skip_reduce": True}], ids=lambda c: next(iter(c))
)
def test_a_triton_gemm_with_more_than_the_product_goes_back_to_aiter(op_calls, changed):
    assert triton_call(**changed) is FallbackReason.OUT_OF_CONTRACT
    assert op_calls == []


def test_a_triton_gemm_reaches_the_op_with_the_callers_buffer(op_calls):
    y = torch.empty(4, 6)
    assert triton_call(y=y, dtype=torch.float16) == "aklr"
    (call,) = op_calls
    assert call["out"] is y
    assert call["out_dtype"] is torch.float16
