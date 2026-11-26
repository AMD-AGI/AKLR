# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on sglang 0.5.24.dev809's sparse_mla_prefill_gfx950 (Kimi-K3.1-Pro's image): what reaches
dsa_sparse_mla_fp8_ds_prefill, in what shapes, and what goes back to sglang; the op is a recorder."""

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.dsa_backend_hip import api

TOKENS, HEADS, TOPK, SLOTS = 3, 2, 8, 16


class _Op:
    """The op: its calls and the calls it was asked about, answered with ``reason`` (None: served)."""

    def __init__(self):
        self.calls, self.asked, self.reason = [], [], None

    def __call__(self, **call):
        self.calls.append(call)
        if self.reason is not None:
            return self.reason
        return torch.full((TOKENS, HEADS, 512), 3.0), torch.full((TOKENS, HEADS), 5.0)

    def fallback_reason(self, **call):
        self.asked.append(call)
        return self.reason


@pytest.fixture
def op(monkeypatch):
    recorder = _Op()
    monkeypatch.setattr(api, "dsa_sparse_mla_fp8_ds_prefill", recorder)
    return recorder


def prefill_call(**kwargs):
    """triton_sparse_mla_fwd's call: q_nope and q_rope views of one q, the pool [slots, 1, 656], indices
    [tokens, 1, topk]."""
    q = torch.randn(TOKENS, HEADS, 576, dtype=torch.bfloat16)
    kv = torch.zeros(SLOTS, 1, 656, dtype=torch.uint8)
    indices = torch.randint(0, SLOTS, (TOKENS, 1, TOPK), dtype=torch.int32)
    call = {"q_nope": q[..., :512], "q_rope": q[..., 512:], "kv": kv, "indices": indices, "sm_scale": 0.07}
    return call, api._sparse_mla_prefill_gfx950(**call, **kwargs)


def test_a_prefill_reaches_the_op_with_its_pool_rows_and_dense_q(op):
    """The pool's rows and the indices are viewed, q's parts copied dense once the op takes them; the
    target's [1, tokens, heads, 512] output and its LSE come back."""
    call, result = prefill_call(return_lse=True)
    ((asked, served),) = zip(op.asked, op.calls, strict=True)
    assert served["kv_cache"].shape == (SLOTS, 656) and served["kv_cache"].data_ptr() == call["kv"].data_ptr()
    assert torch.equal(served["indices"], call["indices"][:, 0])
    assert served["q_nope"].is_contiguous() and torch.equal(served["q_nope"], call["q_nope"])
    assert served["q_rope"].is_contiguous() and torch.equal(served["q_rope"], call["q_rope"])
    assert asked["q_nope"].shape == call["q_nope"].shape and served["sm_scale"] == 0.07
    output, lse = result
    assert output.shape == (1, TOKENS, HEADS, 512) and bool((output == 3).all())
    assert bool((lse == 5).all())


def test_without_return_lse_only_the_output_comes_back(op):
    _, result = prefill_call()
    assert torch.is_tensor(result) and result.shape == (1, TOKENS, HEADS, 512)


def test_a_call_the_op_declines_copies_nothing_and_goes_back_to_sglang(op):
    op.reason = FallbackReason.NO_WORKLOAD
    _, result = prefill_call(return_lse=True)
    assert result is FallbackReason.NO_WORKLOAD
    assert len(op.asked) == 1 and op.calls == []


def test_a_value_width_other_than_512_stays_on_sglang(op):
    _, result = prefill_call(d_v=448)
    assert result is FallbackReason.OUT_OF_CONTRACT
    assert op.asked == [] and op.calls == []
