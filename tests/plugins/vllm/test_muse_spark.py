# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.muse_spark import api


def _experts_apply(hidden_states):
    return api._experts_apply(None, None, hidden_states, *[None] * 13)


def test_rows_without_a_deferred_norm_run_vllm(monkeypatch):
    monkeypatch.setattr(api, "_PENDING", set())
    assert _experts_apply(torch.zeros(2, 4)) is FallbackReason.OUT_OF_CONTRACT


def test_a_deferred_norm_lost_on_the_way_ends_the_step(monkeypatch):
    """Rows other than the pooled ones reach the experts while a norm is left to them: vLLM's
    apply would run on unnormalized rows."""
    monkeypatch.setattr(api, "_PENDING", {1})
    with pytest.raises(RuntimeError, match="other than the pooled"):
        _experts_apply(torch.zeros(2, 4))


@pytest.mark.gpu
@pytest.mark.parametrize("n_tokens", [1, 7, 256])
@pytest.mark.parametrize("gain_center", [0.0, 1.0])
def test_the_baselines_norm_is_vllms(n_tokens, gain_center):
    """The copy of vLLM's norm kernel the baseline runs writes vLLM's bits."""
    fused_norm_expand = pytest.importorskip("vllm.models.muse_spark.amd.ops.fused_norm_expand")
    from aklr.plugins.vllm.models.muse_spark._norm_expand import rmsnorm_expand

    torch.manual_seed(0)
    x = torch.randn(n_tokens, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    first = fused_norm_expand.fused_rmsnorm_expand(x, weight, 1e-6, gain_center, False, 8)
    expected = fused_norm_expand.take_expanded(first, 8)
    assert torch.equal(rmsnorm_expand(x, weight, 1e-6, gain_center, False, 8), expected)
