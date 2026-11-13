# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's aiter-backend MLA prefill to AKLR's mla_prefill_varlen and mla_prefill_varlen_lse: what is taken, what
runs on bf16 flash attention and what goes back to sglang, on plans left unbuilt."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers import aiter_backend_mla_prefill as route

HEADS, QK, V_DIM = 12, 192, 128


class _Op:
    """An AKLR op as the route sees it: its fallback reason, then each call's arguments; (output, lse) with the
    lse as [tokens, heads] for the lse op, the output alone otherwise."""

    def __init__(self, with_lse, reason=None):
        self.with_lse, self.reason, self.asked, self.calls = with_lse, reason, [], []

    def fallback_reason(self, q, k, v, offsets, **kwargs):
        self.asked.append(dict(v_contiguous=v.is_contiguous(), offsets=offsets, **kwargs))
        return self.reason

    def __call__(self, q, k, v, offsets, **kwargs):
        self.calls.append(dict(v_contiguous=v.is_contiguous(), offsets=offsets, **kwargs))
        out = torch.zeros(q.shape[0], HEADS, V_DIM)
        lse = torch.arange(q.shape[0] * HEADS, dtype=torch.float32).view(-1, HEADS)
        return (out, lse) if self.with_lse else out


@pytest.fixture
def ops(monkeypatch):
    ops = SimpleNamespace(varlen=_Op(False), lse=_Op(True))
    monkeypatch.setattr(route, "mla_prefill_varlen", ops.varlen)
    monkeypatch.setattr(route, "mla_prefill_varlen_lse", ops.lse)
    return ops


@pytest.fixture
def flash(monkeypatch):
    """aiter's bf16 flash attention: its calls; the lse it returns is [heads, tokens]."""
    calls = []

    def flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal, return_lse):
        calls.append(dict(causal=causal, return_lse=return_lse, max_k=max_k))
        lse = torch.arange(q.shape[0] * HEADS, dtype=torch.float32).view(q.shape[0], HEADS).t().contiguous()
        return torch.ones(q.shape[0], HEADS, V_DIM), lse, None

    aiter = ModuleType("aiter")
    aiter.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    return calls


def tensors(n_q=300, n_k=300):
    # kv_b's output: v a strided view of k's buffer, as sglang hands it over
    kv = torch.zeros(n_k, HEADS, QK + V_DIM, dtype=torch.bfloat16)
    return torch.zeros(n_q, HEADS, QK, dtype=torch.bfloat16), kv[..., :QK], kv[..., QK:]


def ps_of(n_q=300, n_k=300, need_lse=True, causal=True):
    return SimpleNamespace(
        qo_indptr=torch.tensor([0, n_q], dtype=torch.int32),
        kv_indptr=torch.tensor([0, n_k], dtype=torch.int32),
        max_q_len=n_q,
        is_causal=causal,
        need_lse=need_lse,
    )


LAYER = SimpleNamespace(scaling=0.1)


def test_the_main_prefills_plan_is_sglangs(monkeypatch):
    call = dict(
        qo_indptr=None, kv_indptr=None, kv_lens_cpu=None, num_kv_tokens=0, max_q_len=1, is_causal=True
    )
    assert route._build_prefill_ps_metadata(None, **call, need_lse=False) is FallbackReason.OUT_OF_CONTRACT


def test_the_chunked_prefixs_plans_are_left_unbuilt(monkeypatch):
    backend = ModuleType("sglang.srt.layers.attention.aiter_backend")
    backend.MlaPrefillPsMetadata = lambda **fields: SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    qo, kv = torch.tensor([0, 3]), torch.tensor([0, 9])
    call = dict(qo_indptr=qo, kv_indptr=kv, kv_lens_cpu=None, num_kv_tokens=9, max_q_len=3, is_causal=False)
    ps = route._build_prefill_ps_metadata(None, **call, need_lse=True)
    assert (
        ps.qo_indptr is qo and ps.kv_indptr is kv and ps.max_q_len == 3 and not ps.is_causal and ps.need_lse
    )
    assert all(
        getattr(ps, f) is None for f in ("kv_indices", "work_indptr", "work_info_set", "reduce_indptr")
    )


def test_the_causal_pass_over_the_extend_tokens_reaches_the_lse_op_in_fp8(ops, flash):
    q, k, v = tensors()
    output, lse = route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of())
    (asked,), (c,) = ops.lse.asked, ops.lse.calls
    assert asked["v_contiguous"] and c["v_contiguous"]
    assert c["causal"] is True and c["cu_seqlens_k"] is None and c["qkv_fp8"] is True and c["sm_scale"] == 0.1
    assert lse.shape == (300, HEADS) and flash == [] and ops.varlen.calls == []


def test_a_prefix_chunk_pass_reads_its_keys_without_the_causal_mask(ops, flash):
    q, k, v = tensors(n_k=900)
    output, lse = route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of(n_k=900, causal=False))
    (c,) = ops.lse.calls
    assert c["causal"] is False and c["cu_seqlens_k"].tolist() == [0, 900]
    assert lse is not None


def test_a_declined_lse_pass_runs_on_bf16_flash_attention_never_sglangs_fp8_path(ops, flash):
    ops.lse.reason = FallbackReason.DISABLED
    q, k, v = tensors()
    output, lse = route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of())
    assert flash == [dict(causal=True, return_lse=True, max_k=300)] and ops.lse.calls == []
    # merge_state reads the lse as [tokens, heads]
    assert lse.shape == (300, HEADS) and lse.is_contiguous() and lse[2, 3].item() == 2 * HEADS + 3


def test_offsets_off_the_16_byte_boundary_are_copied(ops, flash):
    q, k, v = tensors()
    ps = ps_of()
    rows = torch.tensor([7, 0, 300, 0], dtype=torch.int32)
    ps.qo_indptr = rows[1:3]  # 4 bytes past the buffer's start
    route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps)
    (c,) = ops.lse.calls
    assert c["offsets"].tolist() == [0, 300] and c["offsets"].data_ptr() % 16 == 0


def test_the_main_prefill_reaches_mla_prefill_varlen_and_returns_no_lse(ops, flash):
    q, k, v = tensors(n_k=900)
    output, lse = route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of(n_k=900, need_lse=False))
    (c,) = ops.varlen.calls
    assert c["cu_seqlens_k"].tolist() == [0, 900] and c["v_contiguous"] and lse is None


def test_the_main_prefill_over_more_keys_than_a_segment_takes_the_segmented_lse_kernel(
    ops, flash, monkeypatch
):
    monkeypatch.setattr(route, "_KV_SEGMENT", 512)
    q, k, v = tensors(n_k=900)
    output, lse = route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of(n_k=900, need_lse=False))
    (c,) = ops.lse.calls
    assert c["causal"] is True and lse is None and ops.varlen.calls == []


def test_a_declined_main_prefill_goes_back_to_sglangs_fp8_path(ops, flash):
    ops.varlen.reason = FallbackReason.NO_WORKLOAD
    q, k, v = tensors()
    assert (
        route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of(need_lse=False))
        is FallbackReason.NO_WORKLOAD
    )
    assert ops.varlen.calls == [] and flash == []


def test_inside_a_graph_capture_no_pass_reaches_aklr(ops, flash, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    q, k, v = tensors()
    assert (
        route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of(need_lse=False))
        is FallbackReason.NOT_WARMED
    )
    route._mla_fp8_prefill_attn_ps(None, q, k, v, LAYER, ps_of())
    assert ops.lse.asked == [] and ops.varlen.asked == [] and len(flash) == 1
