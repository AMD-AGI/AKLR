# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The torch GEMMs sglang runs for Kimi-K3, routed to AKLR's gemm: the DSpark draft's lm_head and K3's own bf16
GEMMs. What each hands the op, and what stays on sglang before the op is asked."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models import dspark_torch_gemm as dspark_route
from aklr.plugins.sglang.models import kimi_k3_torch_gemm as k3_route

ROWS, HIDDEN, VOCAB = 4, 16, 32


class _Gemm:
    """AKLR's gemm as the routes see it: each call's arguments, and what the test sets it to return."""

    def __init__(self):
        self.calls, self.result = [], "aklr"

    def __call__(self, a, b, **kwargs):
        self.calls.append(dict(a=a, b=b, **kwargs))
        return self.result


@pytest.fixture
def gemm(monkeypatch):
    fake = _Gemm()
    for module in (dspark_route, k3_route):
        monkeypatch.setattr(module, "gemm", fake)
    return fake


@pytest.fixture
def sglang(monkeypatch):
    """sglang's helper the DSpark route asks: a head is quantized where it carries ``quantized``; the DSpark module,
    without 0.5.23's fp8 head flag."""
    logits = ModuleType("sglang.srt.layers.logits_processor")
    logits.should_apply_lm_head_quant_method = lambda head, method: getattr(head, "quantized", False)
    dspark = ModuleType("sglang.srt.models.dspark")
    for module in (logits, dspark):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return dspark


def head(**traits):
    return SimpleNamespace(
        weight=torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16), quant_method=None, **traits
    )


def hidden(dtype=torch.bfloat16):
    return torch.zeros(ROWS, HIDDEN, dtype=dtype)


def test_hidden_states_of_another_dtype_reach_gemm_unconverted(sglang, gemm):
    """gemm declines them, sglang converts: nothing is copied before the op is asked."""
    x = hidden(torch.float32)
    dspark_route._project_through_lm_head(x, head())
    assert [call["a"] for call in gemm.calls] == [x]


def test_a_quantized_draft_head_stays_on_sglang(sglang, gemm):
    assert (
        dspark_route._project_through_lm_head(hidden(), head(quantized=True))
        is FallbackReason.OUT_OF_CONTRACT
    )
    assert gemm.calls == []


def test_the_fp8_draft_head_stays_on_sglang(sglang, gemm):
    sglang._DSPARK_FP8_LM_HEAD = True
    assert dspark_route._project_through_lm_head(hidden(), head()) is FallbackReason.OUT_OF_CONTRACT
    assert gemm.calls == []


@pytest.mark.parametrize("result", ["aklr", FallbackReason.NO_WORKLOAD], ids=["served", "declined"])
def test_a_plain_draft_head_hands_its_hidden_states_and_weight_to_gemm(sglang, gemm, result):
    gemm.result = result
    x, lm_head = hidden(), head()
    assert dspark_route._project_through_lm_head(x, lm_head) is result
    (call,) = gemm.calls
    assert call["a"] is x and call["b"] is lm_head.weight


@pytest.mark.parametrize(
    ("out", "out_dtype", "expected"),
    [
        (torch.empty(ROWS, VOCAB, dtype=torch.float32), None, torch.float32),
        (torch.empty(ROWS, VOCAB, dtype=torch.float32), torch.bfloat16, torch.bfloat16),
        (None, None, None),
        (None, torch.float32, torch.float32),
    ],
    ids=["buffer_sets_dtype", "explicit_dtype_wins", "neither", "dtype_alone"],
)
def test_k3s_bf16_gemm_infers_its_output_dtype_from_the_callers_buffer(gemm, out, out_dtype, expected):
    x, weight = hidden(), torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
    k3_route._k3_bf16_gemm(x, weight, out=out, out_dtype=out_dtype)
    (call,) = gemm.calls
    assert call["a"] is x and call["b"] is weight and call["out"] is out and call["out_dtype"] == expected


def test_k3s_tiny_gemm_hands_both_operands_to_gemm(gemm):
    x, weight = hidden(), torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
    assert k3_route._kimi_k3_tiny_gemm(x, weight) == "aklr"
    (call,) = gemm.calls
    assert call["a"] is x and call["b"] is weight
