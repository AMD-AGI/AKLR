# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The attention-residual kernels on finite inputs past float32's squares: rows whose sums of squares overflow
unscaled, and scores below the masked lanes' old sentinel. Each case's rows are equal, so the softmax is uniform and
every normalized value is about one."""

import pytest
import torch

from aklr.core.platform.env import ENV_MIN_SPEEDUP
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.plugins.sglang.layers.attn_residual.api import add_attn_res, attn_res

HIDDEN, BANK_ROWS = 7168, 8

CASES = {
    # values whose squares pass float32 (2**128), score weights 0
    "rows_of_2^64": (2.0**64, 0.0),
    # the unnormalised mix of nine rows of 2**57 squares past float32
    "mix_of_rows_of_2^57": (2.0**57, 0.0),
    # rows of ones scored about -3.2e38, below -3.0e38
    "scores_near_-3.2e38": (1.0, -4.5e34),
}


@pytest.mark.gfx950
# the decode kernels serve 8 to 64 tokens at up to 7 rows; at 8 rows, the prefill kernels any count
@pytest.mark.parametrize("n_tokens", [8, 256, 2048])
@pytest.mark.parametrize("n_valid_rows", [3, 8])
@pytest.mark.parametrize("with_addend", [False, True], ids=["attn_res", "add_attn_res"])
@pytest.mark.parametrize("case", CASES, ids=str)
def test_every_finite_case_normalizes_to_about_one(
    default_dispatchers: DispatcherRegistry,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    with_addend: bool,
    n_valid_rows: int,
    n_tokens: int,
) -> None:
    monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
    default_dispatchers.init()
    value, weight = CASES[case]
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    prefix = torch.full((n_tokens, HIDDEN), value, **bf16)
    bank = torch.full((n_tokens, BANK_ROWS, HIDDEN), value, **bf16)
    score_weight = torch.full((HIDDEN,), weight, dtype=torch.float32, device="cuda")
    norm_weight = torch.ones(HIDDEN, **bf16)
    common = dict(score_eps=1e-6, norm_eps=1e-6, n_valid_rows=n_valid_rows, write_bank=False)
    if with_addend:
        result = add_attn_res(prefix, torch.zeros_like(prefix), bank, score_weight, norm_weight, **common)
    else:
        result = attn_res(prefix, bank, score_weight, norm_weight, **common)
    if isinstance(result, FallbackReason):
        pytest.skip(f"no kernel serves the call: {result.value}")
    normalized = result[0] if isinstance(result, tuple) else result
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(
        normalized.float(), torch.ones_like(normalized, dtype=torch.float32), rtol=1e-2, atol=1e-2
    )
