# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Kimi-K3 DCP target-verify route on stand-ins of sglang 0.5.22's aiter backend: on aiter's
cp schedule, one op call whose per-row lengths make each query row read its rank's shard up to
its own global position; on the gluon schedule, sglang's own path."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.kimi_k3 import api

SIZE, HEADS, Q_LEN = 8, 96, 5
# global lengths, window included: 3 shorter than the ranks, one a rank's shard ends inside the window
LENGTHS = (Q_LEN + 3, 1000, 4096 + 2)


class Op:
    """flash_mla_with_kvcache's api: takes every call, records what it was handed."""

    def __init__(self, ask=None):
        self.ask, self.asked, self.calls = ask, 0, []

    def fallback_reason(self, **kwargs):
        self.asked += 1
        return self.ask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        rows = kwargs["q"].shape[0]
        return torch.zeros(rows, 1, HEADS, 512), torch.zeros(rows, 1, HEADS)


def backend(cprr=True, logit_cap=0.0):
    rows = len(LENGTHS) * Q_LEN
    indptr = torch.tensor((0, *LENGTHS), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata = SimpleNamespace(
        use_asm_cprr_verify=cprr,
        max_q_len=Q_LEN,
        verify_token_table=torch.zeros(rows, 640, dtype=torch.int32),
        local_kv_lens=torch.zeros(rows, dtype=torch.int32),
        global_kv_indptr=indptr,
    )
    pool = torch.zeros(4097, 1, 576, dtype=torch.uint8)
    layer = SimpleNamespace(
        layer_id=3,
        logit_cap=logit_cap,
        tp_q_head_num=HEADS,
        qk_head_dim=576,
        v_head_dim=512,
        scaling=192**-0.5,
    )
    attn = SimpleNamespace(
        forward_metadata=metadata,
        page_size=1,
        dcp_world_size=SIZE,
        kv_cache_dtype=torch.float8_e4m3fn,
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: pool),
        _resolve_fp8_kv_scale_float=lambda layer, k_descale: 1.0,
    )
    return attn, layer, torch.zeros(rows, HEADS * 576, dtype=torch.bfloat16)


@pytest.fixture
def rank(monkeypatch):
    """sglang's runtime context at a settable DCP rank."""
    parallel = SimpleNamespace(attn_dcp_rank=0)
    stand_in = ModuleType("sglang.srt.runtime_context")
    stand_in.get_parallel = lambda: parallel
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", stand_in)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return parallel


@pytest.fixture
def op(monkeypatch):
    stand_in = Op()
    monkeypatch.setattr(api, "flash_mla_with_kvcache", stand_in)
    return stand_in


@pytest.mark.parametrize("dcp_rank", [0, 3, 7])
def test_each_row_reads_its_shard_up_to_its_global_position(rank, op, dcp_rank):
    """Row i of a request of global length L reads the shard's tokens at global positions
    0 .. L - Q_LEN + i on rank p % SIZE, counted here one by one."""
    rank.attn_dcp_rank = dcp_rank
    attn, layer, q = backend()
    out, lse = api._forward_verify_dcp(attn, q, None, layer, None)
    assert out.shape == (q.shape[0], HEADS, 512) and lse.shape == (q.shape[0], HEADS)
    (call,) = op.calls
    expected = [
        sum(1 for p in range(length - Q_LEN + i + 1) if p % SIZE == dcp_rank)
        for length in LENGTHS
        for i in range(Q_LEN)
    ]
    assert call["cache_seqlens"].dtype == torch.int32
    assert call["cache_seqlens"].tolist() == expected
    assert call["block_table"] is attn.forward_metadata.verify_token_table
    assert call["kv_cache"].dtype == torch.float8_e4m3fn and call["kv_cache"].shape == (4097, 1, 1, 576)
    assert call["q"].shape == (q.shape[0], 1, HEADS, 576) and call["return_lse"]


def test_the_lengths_are_computed_once_a_forward(rank, op):
    """Every layer of a forward reads one tensor of lengths."""
    attn, layer, q = backend()
    api._forward_verify_dcp(attn, q, None, layer, None)
    api._forward_verify_dcp(attn, q, None, layer, None)
    first, second = op.calls
    assert first["cache_seqlens"] is second["cache_seqlens"]


def test_a_graph_capture_recomputes_the_warm_up_lengths(rank, op, monkeypatch):
    """sglang warms up on the capture's metadata: the capture must record its own lengths, not
    read the warm-up's tensor, which no replay would update."""
    attn, layer, q = backend()
    api._forward_verify_dcp(attn, q, None, layer, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    api._forward_verify_dcp(attn, q, None, layer, None)
    api._forward_verify_dcp(attn, q, None, layer, None)
    warm_up, captured, next_layer = (call["cache_seqlens"] for call in op.calls)
    assert captured is not warm_up and next_layer is captured


@pytest.mark.parametrize("case", [{"cprr": False}, {"logit_cap": 30.0}], ids=["gluon_schedule", "logit_cap"])
def test_other_schedules_stay_on_sglang(rank, op, case):
    attn, layer, q = backend(**case)
    assert api._forward_verify_dcp(attn, q, None, layer, None) is FallbackReason.OUT_OF_CONTRACT
    assert op.asked == 0 and not op.calls


def test_an_unserved_point_stays_on_sglang(rank, op):
    """A point the op would not serve: sglang's own path, before any length is computed."""
    op.ask = FallbackReason.NO_WORKLOAD
    attn, layer, q = backend()
    assert api._forward_verify_dcp(attn, q, None, layer, None) is FallbackReason.NO_WORKLOAD
    assert op.asked == 1 and not op.calls
    assert not hasattr(attn.forward_metadata, "_aklr_causal_shard_lens")
