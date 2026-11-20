# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention residual on vLLM to AKLR's kimi_k3_attn_res and kimi_k3_attn_res_fp8: what the route takes
and how it hands it over (CPU), and the baseline's copy of vLLM's Triton kernel against vLLM's own (GPU)."""

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.attn_res import api

TOKENS, HIDDEN, ROWS = 4, 256, 4


@pytest.fixture
def op(monkeypatch):
    calls = []

    def fake(name):
        def call(*args, **kwargs):
            calls.append((args, kwargs))
            return name

        return call

    monkeypatch.setattr(api, "kimi_k3_attn_res", fake("aklr"))
    monkeypatch.setattr(api, "kimi_k3_attn_res_fp8", fake("aklr_fp8"))
    return calls


def operands():
    bf16 = dict(dtype=torch.bfloat16)
    return dict(
        prefix=torch.zeros(TOKENS, HIDDEN, **bf16),
        delta=torch.zeros(TOKENS, HIDDEN, **bf16),
        blocks=torch.zeros(TOKENS, ROWS, HIDDEN, **bf16),
        norm_weight=torch.ones(HIDDEN, **bf16),
        qk_weight=torch.ones(HIDDEN, **bf16),
        output_norm_weight=torch.ones(HIDDEN, **bf16),
    )


@pytest.mark.parametrize(
    ("changed", "write_idx"),
    [({"quant_dtype": torch.float8_e4m3fnuz}, 2), ({}, 1)],
    ids=["fnuz_fp8_output", "write_before_the_last_row"],
)
def test_what_no_contract_states_goes_back_to_vllm(op, changed, write_idx):
    x = operands()
    result = api._attn_res(
        x["prefix"], x["delta"], x["blocks"], x["norm_weight"], x["qk_weight"], x["output_norm_weight"], 2,
        write_idx, 1e-6, 1e-5, **changed,
    )  # fmt: skip
    assert result is FallbackReason.OUT_OF_CONTRACT and op == []


@pytest.mark.parametrize(
    ("quant_dtype", "to"), [(None, "aklr"), (torch.float8_e4m3fn, "aklr_fp8")], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("write_idx", [-1, 2], ids=["no_write", "write_row_n"])
@pytest.mark.parametrize("with_optionals", [True, False], ids=["addend_and_norm", "neither"])
def test_a_supported_call_hands_every_operand_over(op, write_idx, with_optionals, quant_dtype, to):
    x = operands()
    delta = x["delta"] if with_optionals else None
    norm = x["output_norm_weight"] if with_optionals else None
    result = api._attn_res(
        x["prefix"], delta, x["blocks"], x["norm_weight"], x["qk_weight"], norm, 2, write_idx, 1e-6, 1e-5,
        quant_dtype=quant_dtype,
    )  # fmt: skip
    assert result == to
    ((args, kwargs),) = op
    assert args[0] is x["prefix"] and args[1] is x["blocks"]
    assert args[2] is x["norm_weight"] and args[3] is x["qk_weight"]
    assert kwargs == dict(
        n_valid_rows=2,
        write_bank=write_idx >= 0,
        addend=delta,
        norm_weight=norm,
        score_eps=1e-6,
        norm_eps=1e-5,
    )


@pytest.mark.gfx950
@pytest.mark.parametrize("n_tokens", [1, 255, 256, 300])
@pytest.mark.parametrize("n_rows", [0, 1, 3])
@pytest.mark.parametrize("variant", ["addend_write_norm", "plain"])
def test_the_baselines_kernel_is_vllms(n_tokens, n_rows, variant):
    """The copy of vLLM's Triton kernel the baseline runs writes vLLM's bits: output, updated prefix and bank."""
    linear = pytest.importorskip("vllm.models.kimi_k3.amd.linear")
    from aklr.plugins.vllm.models.kimi_k3.attn_res._triton import attn_res

    torch.manual_seed(0)
    full = variant == "addend_write_norm"

    def make():
        torch.manual_seed(1)
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        return [
            torch.randn(n_tokens, 7168, **bf16),
            torch.randn(n_tokens, 7168, **bf16) if full else None,
            torch.randn(n_tokens, 8, 7168, **bf16),
            torch.randn(7168, **bf16),
            torch.randn(7168, **bf16),
            torch.randn(7168, **bf16) if full else None,
        ]

    write = n_rows if full else -1
    ours, theirs = make(), make()
    out = attn_res(*ours, n_rows, write, 1e-6, 1e-5)
    expected = linear.attn_res(*theirs, n_rows, write, 1e-6, 1e-5)
    assert torch.equal(out, expected)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[2], theirs[2])
