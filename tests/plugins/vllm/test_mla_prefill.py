# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MLA prefill over cached context to AKLR's mla_prefill_varlen_lse: what is taken and what goes back."""

from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import mla_prefill

HEADS, QK, V_DIM = 12, 192, 128


class _Op:
    """mla_prefill_varlen_lse as the route sees it: each call's arguments; (output, lse as [tokens, heads])."""

    def __init__(self, reason=None):
        self.calls, self.reason = [], reason

    def fallback_reason(self, *args, **kwargs):
        return self.reason

    def __call__(self, q, k, v, offsets, **kwargs):
        self.calls.append(dict(q=q, k=k, v=v, offsets=offsets, **kwargs))
        return torch.zeros(q.shape[0], HEADS, V_DIM), torch.arange(
            q.shape[0] * HEADS, dtype=torch.float32
        ).view(-1, HEADS)


@pytest.fixture
def op(monkeypatch):
    fake = _Op()
    monkeypatch.setattr(mla_prefill, "mla_prefill_varlen_lse", fake)
    return fake


def backend(offsets=(0, 300)):
    loc = torch.tensor(offsets, dtype=torch.int32)
    return SimpleNamespace(scale=0.1, _prefill_metadata=SimpleNamespace(query_start_loc=loc))


def tensors(n_q=300, n_k=300, dtype=torch.bfloat16):
    return (torch.zeros(n_q, HEADS, QK, dtype=dtype), torch.zeros(n_k, HEADS, QK, dtype=dtype),
            torch.zeros(n_k, HEADS, V_DIM, dtype=dtype))  # fmt: skip


@pytest.mark.parametrize(
    "changed",
    [{"return_softmax_lse": False}, {"out": torch.empty(300, HEADS, V_DIM)}, {"output_scale": torch.ones(1)}],
    ids=["no_context", "callers_buffer", "quantized_output"],
)
def test_new_tokens_vllm_keeps(op, changed):
    q, k, v = tensors()
    call = {"return_softmax_lse": True} | changed
    assert mla_prefill._run_prefill_new_tokens(backend(), q, k, v, **call) is FallbackReason.OUT_OF_CONTRACT
    assert op.calls == []


def test_new_tokens_over_context_reach_the_op_causal_in_fp8(op):
    q, k, v = tensors()
    output, lse = mla_prefill._run_prefill_new_tokens(backend(), q, k, v, return_softmax_lse=True)
    (c,) = op.calls
    assert c["qkv_fp8"] is True and c["causal"] is True and c["cu_seqlens_k"] is None and c["sm_scale"] == 0.1
    assert c["offsets"].tolist() == [0, 300] and c["v"].is_contiguous()
    assert output.shape == (300, HEADS, V_DIM)
    # vLLM reads the lse as [heads, tokens]
    assert lse.shape == (HEADS, 300) and lse.is_contiguous() and lse[3, 2].item() == 2 * HEADS + 3


def test_a_context_chunk_reaches_the_op_without_the_causal_mask(op):
    q, k, v = tensors(n_q=300, n_k=900)
    chunk = SimpleNamespace(query_start_loc=torch.tensor([0, 300], dtype=torch.int32),
                            cu_seq_lens=torch.tensor([0, 900], dtype=torch.int32))  # fmt: skip
    output, lse = mla_prefill._run_prefill_context_chunk(backend(), chunk, q, k, v)
    (c,) = op.calls
    assert c["causal"] is False and c["cu_seqlens_k"].tolist() == [0, 900] and c["qkv_fp8"] is True
    assert lse.shape == (HEADS, 300)


def test_a_declined_pass_goes_back_to_vllm(monkeypatch):
    fake = _Op(reason=FallbackReason.NO_WORKLOAD)
    monkeypatch.setattr(mla_prefill, "mla_prefill_varlen_lse", fake)
    q, k, v = tensors()
    assert (
        mla_prefill._run_prefill_new_tokens(backend(), q, k, v, return_softmax_lse=True)
        is FallbackReason.NO_WORKLOAD
    )
    assert fake.calls == []


def test_a_non_bf16_query_goes_back_to_vllm(op):
    q, k, v = tensors(dtype=torch.float16)
    chunk = SimpleNamespace(query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
                            cu_seq_lens=torch.tensor([0, 5], dtype=torch.int32))  # fmt: skip
    assert mla_prefill._run_prefill_context_chunk(backend(), chunk, q, k, v) is FallbackReason.OUT_OF_CONTRACT
    assert op.calls == []


def test_a_decode_steps_small_extend_stays_on_vllm(op):
    q, k, v = tensors(n_q=4, n_k=66048)
    chunk = SimpleNamespace(query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                            cu_seq_lens=torch.tensor([0, 66048], dtype=torch.int32))  # fmt: skip
    assert mla_prefill._run_prefill_context_chunk(backend(), chunk, q, k, v) is FallbackReason.OUT_OF_CONTRACT
    assert mla_prefill._run_prefill_new_tokens(backend((0, 4)), q, q, v[:4], return_softmax_lse=True) is (
        FallbackReason.OUT_OF_CONTRACT
    )
    assert op.calls == []
