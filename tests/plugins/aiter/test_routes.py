# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's calls to AKLR ops: what is taken, what goes back to aiter, and SiTUv2's activations."""

import enum
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import aiter as aklr_aiter

# the package's names are the functions; the modules hold what the tests replace
moe_module = importlib.import_module("aklr.plugins.aiter.fused_moe")


class ActivationType(enum.Enum):
    Silu = 0
    Situv2 = 1


FP8, FP4 = torch.float8_e4m3fn, torch.float4_e2m1fn_x2


@pytest.fixture
def op_calls(monkeypatch):
    """The AKLR op calls, each as its keyword arguments."""
    calls = []

    def op(*args, **kwargs):
        calls.append(kwargs)
        return "aklr"

    monkeypatch.setattr(moe_module, "aklr_fused_moe", op)
    return calls


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
        moe_module._takes_gate_up_interleave.cache_clear()

    picked.install = install
    install(False)
    monkeypatch.setitem(sys.modules, "aiter", package)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", module)
    yield picked
    moe_module._takes_gate_up_interleave.cache_clear()


NONE_TAKEN = {
    "expert_mask": None,
    "quant_type": None,
    "doweight_stage1": False,
    "w1_scale": None,
    "w2_scale": None,
    "a1_scale": None,
    "a2_scale": None,
    "block_size_M": None,
    "num_local_tokens": None,
    "dtype": None,
    "hidden_pad": 0,
    "intermediate_pad": 0,
    "bias1": None,
    "bias2": None,
    "swiglu_limit": None,
    "beta": None,
    "linear_beta": None,
    "gate_mode": "separated",
    "shared_w1": None,
    "shared_w2": None,
    "shared_expert_id": -1,
    "stage2_scatter": None,
    "output": None,
    "quant_type_a": None,
    "quant_dtype_a": None,
    "quant_dtype_a2": None,
}


def moe_call(activation, gate_up_interleave=None, **changed):
    hidden = torch.zeros(5, 8)
    w1 = torch.zeros(2, 16, 8, dtype=torch.uint8).view(FP4)
    if gate_up_interleave is not None:
        w1.gate_up_interleave = gate_up_interleave
    return aklr_aiter.fused_moe(
        hidden, w1, torch.zeros(2, 8, 8), torch.zeros(5, 2), torch.zeros(5, 2), activation=activation,
        **(NONE_TAKEN | changed),
    )  # fmt: skip


@pytest.mark.parametrize(
    "changed",
    [
        {"block_size_M": 32},
        {"hidden_pad": 128},
        {"intermediate_pad": 128},
        {"a1_scale": torch.ones(1)},
        {"bias1": torch.ones(1)},
        {"shared_w1": torch.ones(1)},
        {"shared_expert_id": 3},
        {"stage2_scatter": object()},
        {"quant_dtype_a": FP8},
        {"herd_topk": 16},
    ],
    ids=lambda changed: next(iter(changed)),
)
def test_what_the_op_does_not_take_goes_back_to_aiter(op_calls, changed):
    assert moe_call(ActivationType.Silu, **changed) is FallbackReason.OUT_OF_CONTRACT
    assert op_calls == []


def test_any_other_activation_keeps_4_bit_activations(op_calls, aiter_picks):
    assert moe_call(ActivationType.Silu) == "aklr"
    assert op_calls[0]["act_quant"] == 4
    assert aiter_picks.calls == []


def test_situv2_takes_fp8_activations_where_aiter_quantizes_to_fp8(op_calls, aiter_picks):
    assert moe_call(ActivationType.Situv2) == "aklr"
    assert op_calls[0]["act_quant"] == 8
    assert aiter_picks.calls == [{"M": 5, "q_dtype_w": FP4}]


@pytest.mark.parametrize("dtype", [FP4, torch.bfloat16], ids=["fp4 unmarked", "bf16"])
def test_situv2_goes_back_to_aiter_on_other_activations(op_calls, aiter_picks, dtype):
    """fp4 activations on weights aiter did not mark as gate-up separated: sglang's aiter marks none."""
    aiter_picks.dtype = dtype
    assert moe_call(ActivationType.Situv2) is FallbackReason.OUT_OF_CONTRACT
    assert op_calls == []


@pytest.mark.parametrize(
    "changed",
    [{"gate_up_interleave": True}, {"gate_mode": "interleave"}, {"expert_mask": torch.ones(2)}],
    ids=lambda changed: next(iter(changed)),
)
def test_fp4_activations_on_anything_else_go_back_to_aiter(op_calls, aiter_picks, changed):
    aiter_picks.dtype = FP4
    marked = {"gate_up_interleave": False} | changed
    assert moe_call(ActivationType.Situv2, **marked) is FallbackReason.OUT_OF_CONTRACT
    assert op_calls == []


def test_an_aiter_that_reads_the_gate_up_interleave_is_asked_with_it(op_calls, aiter_picks):
    aiter_picks.install(True)
    moe_call(ActivationType.Situv2)
    assert aiter_picks.calls == [{"M": 5, "q_dtype_w": FP4, "gate_up_interleave": None}]


def test_the_callers_buffer_and_dtype_reach_the_op(op_calls):
    out = torch.empty(5, 8)
    moe_call(ActivationType.Silu, output=out, dtype=torch.bfloat16, swiglu_limit=0.0)
    (call,) = op_calls
    assert call["out"] is out
    assert call["out_dtype"] is torch.bfloat16
    # vLLM's 0.0 is no limit
    assert call["swiglu_limit"] is None
