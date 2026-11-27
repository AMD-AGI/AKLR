# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's GEMMs that go to aiter's (CPU): its skinny GEMM and Muse Spark's pooler linear call aiter's
gemm_a16w16, which AKLR's aiter provider answers; without the provider vLLM keeps its own."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.plugins.vllm.layers.utils import api as utils
from aklr.plugins.vllm.models.muse_spark import api as muse_spark

A, B, BIAS = torch.zeros(16, 8), torch.zeros(4, 8), torch.zeros(4)
ROUTES = {
    "wv_split_krc": (utils, lambda: utils._wv_split_krc(A, B, 80, BIAS)),
    "muse_spark_linear": (
        muse_spark,
        lambda: muse_spark._torch_linear(None, SimpleNamespace(weight=B), A, BIAS),
    ),
}


@pytest.fixture
def aiter_calls(monkeypatch):
    calls = []
    aiter, tuned_gemm = ModuleType("aiter"), ModuleType("aiter.tuned_gemm")
    aiter.__path__ = []
    tuned_gemm.gemm_a16w16 = lambda *args: calls.append(args) or "aiter"
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", tuned_gemm)
    return calls


@pytest.mark.parametrize("name", ROUTES)
def test_the_route_calls_aiters_gemm(aiter_calls, name):
    module, call = ROUTES[name]
    assert call() == "aiter"
    ((a, b, bias),) = aiter_calls
    assert a is A and b is B and bias is BIAS
