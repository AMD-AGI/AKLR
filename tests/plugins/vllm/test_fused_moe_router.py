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
    monkeypatch.setattr(route, "_REASONS", {})
    monkeypatch.setattr(
        route, "_REMEMBER", True
    )  # read from AKLR_EXPORT at import, which conftest clears later
    # vLLM's own kernel rejects the call, as it does Kimi-K3's router from 10 tokens on
    monkeypatch.setattr(route, "_native_can_use", lambda x, w: False)


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


def test_a_call_vllms_own_kernel_takes_stays_with_it(monkeypatch):
    """vLLM's Triton kernel (at most 9 tokens of Kimi-K3's router): AKLR's gemm is not qualified against it."""
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    monkeypatch.setattr(route, "_native_can_use", lambda x, w: True)
    x, w = tensors(4)
    assert isinstance(route._can_use_rocm_fp32_router_gemm(x, w), FallbackReason)
    assert isinstance(route._rocm_fp32_router_gemm(x, w), FallbackReason)
    assert gemm.asked == [] and gemm.calls == []


def test_a_call_aklr_declines_after_all_runs_what_vllm_runs_without_its_kernel(monkeypatch):
    """A remembered yes that gemm then declines must not reach vLLM's kernel, which rejects the call."""
    monkeypatch.setattr(route, "gemm", _Gemm(FallbackReason.OUT_OF_CONTRACT))
    unrouted = []
    monkeypatch.setattr(route, "_unrouted", lambda x, w: unrouted.append((x, w)) or "vllm")
    x, w = tensors()
    assert route._rocm_fp32_router_gemm(x, w) == "vllm"
    assert unrouted == [(x, w)]


def test_vllms_path_without_its_kernel_for_an_fp32_weight():
    x, w = torch.randn(3, 64, dtype=torch.bfloat16), torch.randn(8, 64)
    out = route._unrouted(x, w)
    assert out.dtype == torch.float32 and torch.equal(out, torch.nn.functional.linear(x.float(), w))


def test_a_call_aklr_serves_takes_vllms_fp32_kernel_path(monkeypatch):
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    x, w = tensors()
    assert route._can_use_rocm_fp32_router_gemm(x, w) is True
    ((args, kwargs),) = gemm.asked
    assert args[0] is x and args[1] is w and kwargs == {"out_dtype": torch.float32}


def test_a_call_aklr_declines_gets_vllms_own_answer_with_gemms_reason(monkeypatch):
    monkeypatch.setattr(route, "gemm", _Gemm(FallbackReason.NO_WORKLOAD))
    assert route._can_use_rocm_fp32_router_gemm(*tensors()) is FallbackReason.NO_WORKLOAD


def test_the_answer_is_asked_once_per_call_shape(monkeypatch):
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    _, w = tensors()
    for n in (16, 16, 4632, 16):
        assert route._can_use_rocm_fp32_router_gemm(tensors(n)[0], w) is True
    assert len(gemm.asked) == 2


def test_a_misaligned_view_of_a_served_shape_is_asked_anew(monkeypatch):
    """gemm declines a pointer off its 16-byte alignment, so a same-shaped view does not share the answer."""
    gemm = _Gemm(None)
    monkeypatch.setattr(route, "gemm", gemm)
    x, w = tensors()
    offset = torch.zeros(16 * 7168 + 1, dtype=torch.bfloat16)[1:].view(16, 7168)
    assert offset.shape == x.shape and offset.stride() == x.stride()
    route._can_use_rocm_fp32_router_gemm(x, w)
    route._can_use_rocm_fp32_router_gemm(offset, w)
    assert len(gemm.asked) == 2


def test_a_not_warmed_answer_is_asked_again(monkeypatch):
    """Inside a graph capture gemm answers not-warmed until the call has run outside one."""
    gemm = _Gemm(FallbackReason.NOT_WARMED)
    monkeypatch.setattr(route, "gemm", gemm)
    x, w = tensors()
    assert route._can_use_rocm_fp32_router_gemm(x, w) is FallbackReason.NOT_WARMED
    gemm.reason = None
    assert route._can_use_rocm_fp32_router_gemm(x, w) is True
    assert len(gemm.asked) == 2
