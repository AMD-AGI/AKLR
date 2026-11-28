# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on sglang 0.5.24.dev809's k3_a8w4_moe (Kimi-K3.1-Pro's image): which calls reach
kimi_k3_routed_moe_mxfp8, in what dtypes, and what goes back to sglang; the op is a recorder."""

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.kimi_k3 import api as kimi_k3

TOKENS, LATENT = 4, 512


@pytest.fixture
def op(monkeypatch):
    """The op's calls, answered with ``answer[0]``: the given ``out``, or 3s of its shape."""
    calls, answer = [], [None]

    def record(**call):
        calls.append(call)
        if answer[0] is not None:
            return answer[0]
        return call["out"] if call["out"] is not None else torch.full((TOKENS, LATENT), 3.0)

    monkeypatch.setattr(kimi_k3, "kimi_k3_routed_moe_mxfp8", record)
    return calls, answer


def experts_call(**kwargs):
    """k3_a8w4_moe's call as Kimi-K3.1's MoE makes it, with stand-in weights."""
    ids = torch.zeros(TOKENS, 16, dtype=torch.int32)
    weights = torch.zeros(TOKENS, 16)
    weight = torch.zeros(1, dtype=torch.uint8)
    hidden = torch.zeros(1, 1, dtype=torch.bfloat16).expand(TOKENS, LATENT)
    return kimi_k3._k3_a8w4_moe(
        hidden, ids, weights, weight, weight, weight, weight, beta=4.0, linear_beta=25.0, **kwargs
    )


def mxfp8_input():
    return (
        torch.zeros(TOKENS, LATENT, dtype=torch.uint8),
        torch.full((TOKENS, LATENT // 32), 127, dtype=torch.uint8),
    )


def test_a_bf16_input_stays_on_sglang(op):
    """Without a_mx the experts quantize hidden_states themselves: not the op's input."""
    calls, _ = op
    out = torch.empty(TOKENS, LATENT, dtype=torch.bfloat16)
    assert experts_call(out=out) is FallbackReason.OUT_OF_CONTRACT
    assert calls == []


def test_the_mxfp8_input_reaches_the_op_as_e4m3_and_e8m0(op):
    """a_mx's two uint8 tensors are the op's a and a_scale, viewed, not copied; the uint8 weights are
    MXFP4 marked shuffled, their scales e8m0, viewed too; the output bf16; the result is out."""
    calls, _ = op
    a, a_scale = mxfp8_input()
    out = torch.empty(TOKENS, LATENT, dtype=torch.bfloat16)
    assert experts_call(out=out, a_mx=(a, a_scale)) is out
    (call,) = calls
    assert (call["a"].dtype, call["a_scale"].dtype) == (torch.float8_e4m3fn, torch.float8_e8m0fnu)
    assert call["a"].data_ptr() == a.data_ptr() and call["a_scale"].data_ptr() == a_scale.data_ptr()
    assert {call[w].dtype for w in ("w1", "w2")} == {torch.float4_e2m1fn_x2}
    assert call["w1"].is_shuffled and call["w2"].is_shuffled
    assert {call[s].dtype for s in ("w1_scale", "w2_scale")} == {torch.float8_e8m0fnu}
    assert (call["beta"], call["linear_beta"]) == (4.0, 25.0)
    assert call["out_dtype"] == torch.bfloat16 and call["out"] is out


def test_the_call_as_sglang_makes_it_is_a_point_of_the_op(op):
    """k3_a8w4_moe's weights and scales as sglang holds them, uint8 in aiter's gate-up-interleaved
    preshuffle (the scales 2-D, w2's padded to 8 columns): the op's call has a FastKey, in that layout."""
    from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
    from aklr.plugins.sglang.models.kimi_k3.contract import KIMI_K3_ROUTED_MOE_MXFP8

    calls, _ = op
    experts, inter = 16, 128
    weights = {
        "w13": torch.zeros(experts, 2 * inter, LATENT // 2, dtype=torch.uint8),
        "w13_scale": torch.zeros(experts * 2 * inter, LATENT // 32, dtype=torch.uint8),
        "w2": torch.zeros(experts, LATENT, inter // 2, dtype=torch.uint8),
        "w2_scale": torch.zeros(experts * LATENT, 8, dtype=torch.uint8),
    }
    hidden = torch.zeros(1, 1, dtype=torch.bfloat16).expand(TOKENS, LATENT)
    routes = (torch.arange(16, dtype=torch.int32).repeat(TOKENS, 1), torch.zeros(TOKENS, 16))
    out = torch.empty(TOKENS, LATENT, dtype=torch.bfloat16)
    kimi_k3._k3_a8w4_moe(hidden, *routes, **weights, beta=4.0, linear_beta=25.0, out=out, a_mx=mxfp8_input())
    (call,) = calls
    codec = FastKeyCodec.from_contract(KIMI_K3_ROUTED_MOE_MXFP8)
    fast_key = FastKeyFnGenerator(codec, None).generate()
    key = fast_key({p.name: call.get(p.name, p.default) for p in KIMI_K3_ROUTED_MOE_MXFP8.api_params})
    assert key is not None
    assert codec.to_point(key).layout_variant_name == "gate_up_interleaved"


def test_a_row_strided_out_is_written_from_the_ops_own(op):
    """A view of rows wider than the latent: the op writes its own output, which lands in the view only."""
    calls, _ = op
    rows = torch.zeros(TOKENS, LATENT + 64, dtype=torch.bfloat16)
    out = rows[:, :LATENT]
    assert experts_call(out=out, a_mx=mxfp8_input()) is out
    assert calls[0]["out"] is None
    assert bool((out == 3).all()) and bool((rows[:, LATENT:] == 0).all())


def test_a_call_the_op_declines_goes_back_to_sglang(op):
    calls, answer = op
    answer[0] = FallbackReason.NO_WORKLOAD
    rows = torch.zeros(TOKENS, LATENT + 64, dtype=torch.bfloat16)
    assert experts_call(out=rows[:, :LATENT], a_mx=mxfp8_input()) is FallbackReason.NO_WORKLOAD
    assert bool((rows == 0).all())
