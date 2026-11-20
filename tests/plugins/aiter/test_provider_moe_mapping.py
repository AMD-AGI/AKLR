# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The aiter provider's fused MoE mapping (``op_kwargs``): what AKLR's fused_moe takes of a call, what goes
back to aiter, and the activations SiTUv2 and SwiGLU are taken at. What the selector declines before the
mapping is in test_provider_moe."""

import enum
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason

moe = importlib.import_module("aklr.plugins.aiter.provider.moe")


class ActivationType(enum.Enum):
    Silu = 0
    Situv2 = 1
    Swiglu = 2


FP8, FP4 = torch.float8_e4m3fn, torch.float4_e2m1fn_x2


@pytest.fixture
def aiter_picks(monkeypatch):
    """A stand-in aiter whose fused MoE quantizes activations to what the test sets; the
    keyword arguments of each ``resolve_activation_dtype`` call are kept."""
    picked = SimpleNamespace(dtype=FP8, calls=[], takes_gate_up_interleave=False)
    package = ModuleType("aiter")
    package.dtypes = SimpleNamespace(fp8=FP8, fp4x2=FP4)
    module = ModuleType("aiter.fused_moe")

    def resolve(quant_type, q_dtype_w, *, activation, gate_mode, M, hidden_dtype, has_a1_scale):
        picked.calls.append({"M": M, "q_dtype_w": q_dtype_w})
        return picked.dtype

    def resolve_new(
        quant_type,
        q_dtype_w,
        *,
        activation,
        gate_mode,
        M,
        hidden_dtype,
        has_a1_scale,
        gate_up_interleave=None,
    ):
        picked.calls.append({"M": M, "q_dtype_w": q_dtype_w, "gate_up_interleave": gate_up_interleave})
        return picked.dtype

    def install(new):
        module.resolve_activation_dtype = resolve_new if new else resolve
        moe._takes_gate_up_interleave.cache_clear()

    picked.install = install
    install(False)
    monkeypatch.setitem(sys.modules, "aiter", package)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", module)
    yield picked
    moe._takes_gate_up_interleave.cache_clear()


NONE_TAKEN = {
    "expert_mask": None,
    "quant_type": None,
    "doweight_stage1": False,
    "w1_scale": None,
    "w2_scale": None,
    "num_local_tokens": None,
    "dtype": None,
    "swiglu_limit": None,
    "beta": None,
    "linear_beta": None,
    "gate_mode": "separated",
    "output": None,
}


def mapped(activation, gate_up_interleave=None, rows=5, **changed):
    """``op_kwargs`` of a call on ``rows`` tokens: the op's keyword arguments, or why aiter computes it."""
    hidden = torch.zeros(rows, 8)
    w1 = torch.zeros(2, 16, 8, dtype=torch.uint8).view(FP4)
    if gate_up_interleave is not None:
        w1.gate_up_interleave = gate_up_interleave
    return moe.op_kwargs(hidden, w1, activation=activation, **(NONE_TAKEN | changed))


def test_any_other_activation_keeps_4_bit_activations(aiter_picks):
    assert mapped(ActivationType.Silu)["act_quant"] == 4
    assert aiter_picks.calls == []


def test_situv2_takes_fp8_activations_where_aiter_quantizes_to_fp8(aiter_picks):
    assert mapped(ActivationType.Situv2)["act_quant"] == 8
    assert aiter_picks.calls == [{"M": 5, "q_dtype_w": FP4}]


@pytest.mark.parametrize("dtype", [FP4, torch.bfloat16], ids=["fp4 unmarked", "bf16"])
def test_situv2_goes_back_to_aiter_on_other_activations(aiter_picks, dtype):
    """fp4 activations on weights aiter did not mark as gate-up separated: sglang's aiter marks none."""
    aiter_picks.dtype = dtype
    assert mapped(ActivationType.Situv2) is FallbackReason.OUT_OF_CONTRACT


@pytest.mark.parametrize(
    "changed",
    [{"gate_up_interleave": True}, {"gate_mode": "interleave"}, {"expert_mask": torch.ones(2)}],
    ids=lambda changed: next(iter(changed)),
)
def test_fp4_activations_on_anything_else_go_back_to_aiter(aiter_picks, changed):
    aiter_picks.dtype = FP4
    marked = {"gate_up_interleave": False} | changed
    assert mapped(ActivationType.Situv2, **marked) is FallbackReason.OUT_OF_CONTRACT


@pytest.mark.parametrize(
    ("rows", "picked", "served"),
    [(255, torch.bfloat16, True), (256, FP4, True), (255, FP4, False), (256, torch.bfloat16, False)],
    ids=["unrounded below 256", "fp4 from 256", "aiter rounds below 256", "aiter keeps from 256"],
)
def test_swiglu_reaches_the_op_only_where_aiter_rounds_as_fused_moe_does(aiter_picks, rows, picked, served):
    """GPTOSS_SWIGLU_MXFP4_BF16_BOUND other than 256 moves the row count from which aiter rounds to fp4."""
    aiter_picks.dtype = picked
    result = mapped(ActivationType.Swiglu, rows=rows)
    assert aiter_picks.calls == [{"M": rows, "q_dtype_w": FP4}]
    if served:
        assert result["act_quant"] == 4
    else:
        assert result is FallbackReason.OUT_OF_CONTRACT


@pytest.mark.parametrize(("bound", "served"), [(256, True), (512, False)])
def test_on_an_aiter_without_the_resolver_swiglu_follows_its_bound(aiter_picks, bound, served):
    del sys.modules["aiter.fused_moe"].resolve_activation_dtype
    sys.modules["aiter.fused_moe"]._SWIGLU_MXFP4_BF16_BOUND = bound
    result = mapped(ActivationType.Swiglu, rows=300)
    assert (result is not FallbackReason.OUT_OF_CONTRACT) is served


def test_swiglu_on_interleaved_gate_and_up_stays_unrounded_without_asking_aiter(aiter_picks):
    assert mapped(ActivationType.Swiglu, rows=300, gate_mode="interleave")["act_quant"] == 4
    assert aiter_picks.calls == []


def test_an_aiter_that_reads_the_gate_up_interleave_is_asked_with_it(aiter_picks):
    aiter_picks.install(True)
    mapped(ActivationType.Situv2)
    assert aiter_picks.calls == [{"M": 5, "q_dtype_w": FP4, "gate_up_interleave": None}]


def test_the_callers_buffer_and_dtype_reach_the_op():
    out = torch.empty(5, 8)
    call = mapped(ActivationType.Silu, output=out, dtype=torch.bfloat16, swiglu_limit=0.0)
    assert call["out"] is out
    assert call["out_dtype"] is torch.bfloat16
    # vLLM's 0.0 is no limit
    assert call["swiglu_limit"] is None
