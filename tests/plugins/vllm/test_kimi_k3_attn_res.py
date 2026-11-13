# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention residual on vLLM to AKLR's kimi_k3_attn_res: what the route takes and how it hands it over
(CPU); the baseline's copy of vLLM's Triton kernel against vLLM's own, and the kernels on finite inputs past float32's
squares (GPU)."""

import pytest
import torch

from aklr.core.platform.env import ENV_MIN_SPEEDUP
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.plugins.vllm.models.kimi_k3.attn_res import api

TOKENS, HIDDEN, ROWS = 4, 256, 4


@pytest.fixture
def op(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return "aklr"

    monkeypatch.setattr(api, "kimi_k3_attn_res", fake)
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
    [({"quant_dtype": torch.float8_e4m3fn}, 2), ({}, 1)],
    ids=["fp8_output", "write_before_the_last_row"],
)
def test_what_no_contract_states_goes_back_to_vllm(op, changed, write_idx):
    x = operands()
    result = api._attn_res(
        x["prefix"], x["delta"], x["blocks"], x["norm_weight"], x["qk_weight"], x["output_norm_weight"], 2,
        write_idx, 1e-6, 1e-5, **changed,
    )  # fmt: skip
    assert result is FallbackReason.OUT_OF_CONTRACT and op == []


@pytest.mark.parametrize("write_idx", [-1, 2], ids=["no_write", "write_row_n"])
@pytest.mark.parametrize("with_optionals", [True, False], ids=["addend_and_norm", "neither"])
def test_a_supported_call_hands_every_operand_over(op, write_idx, with_optionals):
    x = operands()
    delta = x["delta"] if with_optionals else None
    norm = x["output_norm_weight"] if with_optionals else None
    result = api._attn_res(
        x["prefix"], delta, x["blocks"], x["norm_weight"], x["qk_weight"], norm, 2, write_idx, 1e-6, 1e-5
    )
    assert result == "aklr"
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


CASES = {
    # values whose squares pass float32 (2**128), score weights 0
    "rows_of_2^64": (2.0**64, 0.0),
    # the unnormalised mix of nine rows of 2**57 squares past float32
    "mix_of_rows_of_2^57": (2.0**57, 0.0),
    # without a norm, nine rows of 2**125 summed before the softmax denominator divides them
    "rows_of_2^125_no_norm": (2.0**125, 0.0),
}


@pytest.mark.gfx950
@pytest.mark.parametrize("n_tokens", [3, 12])
@pytest.mark.parametrize("n_rows", [1, 3, 8])
@pytest.mark.parametrize("case", CASES, ids=str)
def test_every_finite_case_has_its_finite_result(
    default_dispatchers: DispatcherRegistry,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    n_rows: int,
    n_tokens: int,
) -> None:
    """Rows all equal, so the softmax is uniform: about one with an output norm, the rows' value without."""
    monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
    default_dispatchers.init()
    value, weight = CASES[case]
    with_norm = not case.endswith("no_norm")
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    prefix = torch.full((n_tokens, 7168), value, **bf16)
    bank = torch.full((n_tokens, 8, 7168), value, **bf16)
    # with a zero addend: the schemas without an output norm all take one
    result = api.kimi_k3_attn_res(
        prefix, bank, torch.full((7168,), weight, **bf16), torch.ones(7168, **bf16), n_valid_rows=n_rows,
        addend=torch.zeros_like(prefix), norm_weight=torch.ones(7168, **bf16) if with_norm else None,
        score_eps=1e-6, norm_eps=1e-6,
    )  # fmt: skip
    if isinstance(result, FallbackReason):
        pytest.skip(f"no kernel serves the call: {result.value}")
    out = result[0] if isinstance(result, tuple) else result
    expected = torch.ones_like(out, dtype=torch.float32) * (1.0 if with_norm else value)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
