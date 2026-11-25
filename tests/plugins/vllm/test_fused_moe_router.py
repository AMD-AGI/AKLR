# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's fp32 router GEMM on ROCm: the call that goes to aiter's GEMM, and the answer to vLLM's can-use
question (CPU)."""

import sys
from types import ModuleType

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import fused_moe_router as route


@pytest.fixture(autouse=True)
def aiter_calls(monkeypatch):
    """aiter's gemm_a16w16 faked: its calls. AKLR's aiter provider on, and vLLM's own kernel rejecting the
    call, as it does Kimi-K3's router from 10 tokens on."""
    calls = []

    def gemm_a16w16(a, b, bias=None, otype=None):
        calls.append((a, b, bias, otype))
        return torch.ones(a.shape[0], b.shape[0], dtype=otype)

    aiter, tuned_gemm = ModuleType("aiter"), ModuleType("aiter.tuned_gemm")
    aiter.__path__ = []
    tuned_gemm.gemm_a16w16 = gemm_a16w16
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", tuned_gemm)
    monkeypatch.setattr(route, "_IS_PROVIDER_ON", True)
    monkeypatch.setattr(route, "_native_can_use", lambda x, w: False)
    return calls


def tensors(n=16, dtype=torch.bfloat16):
    return torch.zeros(n, 7168, dtype=torch.bfloat16), torch.zeros(896, 7168, dtype=dtype)


def test_the_router_gemm_calls_aiters_gemm_with_an_fp32_output(aiter_calls):
    x, w = tensors()
    assert route._can_use_rocm_fp32_router_gemm(x, w) is True
    out = route._rocm_fp32_router_gemm(x, w)
    ((a, b, bias, otype),) = aiter_calls
    assert a is x and b is w and bias is None and otype == torch.float32
    assert out.shape == (16, 896) and out.dtype == torch.float32


def test_a_call_vllms_own_kernel_takes_stays_with_it(monkeypatch, aiter_calls):
    """vLLM's Triton kernel (at most 9 tokens of Kimi-K3's router)."""
    monkeypatch.setattr(route, "_native_can_use", lambda x, w: True)
    x, w = tensors(4)
    assert route._can_use_rocm_fp32_router_gemm(x, w) is FallbackReason.NOT_QUALIFIED
    assert route._rocm_fp32_router_gemm(x, w) is FallbackReason.NOT_QUALIFIED
    assert aiter_calls == []


def test_without_aklrs_aiter_provider_vllm_keeps_its_own(monkeypatch, aiter_calls):
    monkeypatch.setattr(route, "_IS_PROVIDER_ON", False)
    x, w = tensors()
    assert route._can_use_rocm_fp32_router_gemm(x, w) is FallbackReason.DISABLED
    assert route._rocm_fp32_router_gemm(x, w) is FallbackReason.DISABLED
    assert aiter_calls == []


def test_an_fp32_router_weight_stays_with_vllm(aiter_calls):
    x, w = tensors(dtype=torch.float32)
    assert route._can_use_rocm_fp32_router_gemm(x, w) is FallbackReason.OUT_OF_CONTRACT
    assert aiter_calls == []
