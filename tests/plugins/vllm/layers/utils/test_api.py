# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's skinny GEMMs to AKLR's gemm: which operand is the rows and which the weight."""

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers.utils import api


@pytest.mark.parametrize(
    "result",
    [torch.ones(5, 48, dtype=torch.bfloat16), FallbackReason.NO_WORKLOAD],
    ids=["served", "fallback"],
)
def test_a_gemm_of_up_to_5_rows_is_asked_for_as_the_rows_times_the_weight(monkeypatch, result):
    """vLLM hands wvSplitK the weight first, then the rows (wvSplitKrc takes them the other way)."""
    calls = []
    monkeypatch.setattr(api, "gemm", lambda a, b, **call: calls.append((a, b, call)) or result)
    weight, rows = torch.zeros(48, 64, dtype=torch.bfloat16), torch.zeros(5, 64, dtype=torch.bfloat16)
    bias = torch.zeros(48, dtype=torch.bfloat16)
    assert api._wv_split_k(weight, rows, 256, bias) is result
    ((a, b, call),) = calls
    assert a is rows and b is weight and call == {"bias": bias}
