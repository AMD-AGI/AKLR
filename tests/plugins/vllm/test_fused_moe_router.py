# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's fp32 router GEMM on ROCm: the call AKLR serves and AKLR's answer to vLLM's can-use question (CPU)."""

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import fused_moe_router as route


class _Gemm:
    """gemm as the route sees it: each question and call; ``reason`` answers both."""

    def __init__(self, reason):
        self.reason, self.asked, self.calls = reason, [], []

    def fallback_reason(self, *args, **kwargs):
        self.asked.append((args, kwargs))
        return self.reason

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.reason or torch.ones(args[0].shape[0], args[1].shape[0], dtype=torch.float32)


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    monkeypatch.setattr(route, "_SERVED", {})


def tensors(n=16):
    return torch.zeros(n, 7168, dtype=torch.bfloat16), torch.zeros(896, 7168, dtype=torch.bfloat16)


def test_the_router_gemm_reaches_gemm_with_an_fp32_output(monkeypatch):
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    x, w = tensors()
    out = route._rocm_fp32_router_gemm(x, w)
    ((args, kwargs),) = gemm.calls
    assert args[0] is x and args[1] is w and kwargs == {"out_dtype": torch.float32}
    assert out.shape == (16, 896) and out.dtype == torch.float32


def test_a_call_aklr_serves_takes_vllms_fp32_kernel_path(monkeypatch):
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    x, w = tensors()
    assert route._can_use_rocm_fp32_router_gemm(x, w) is True
    ((args, kwargs),) = gemm.asked
    assert args[0] is x and args[1] is w and kwargs == {"out_dtype": torch.float32}


def test_a_call_aklr_declines_gets_vllms_own_answer(monkeypatch):
    monkeypatch.setattr(route, "gemm", _Gemm(FallbackReason.NO_WORKLOAD))
    assert isinstance(route._can_use_rocm_fp32_router_gemm(*tensors()), FallbackReason)


def test_the_answer_is_asked_once_per_call_shape(monkeypatch):
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    _, w = tensors()
    for n in (16, 16, 4632, 16):
        assert route._can_use_rocm_fp32_router_gemm(tensors(n)[0], w) is True
    assert len(gemm.asked) == 2
