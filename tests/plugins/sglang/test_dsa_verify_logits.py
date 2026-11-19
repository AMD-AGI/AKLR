# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's logits route under DCP with speculative decoding (CPU): only rows the paged top-k
noted as a target verify's are regrouped by request; a decode batch whose row count the draft tokens
divide stays on the row op."""

import sys
import types

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.dsa_backend import api

NEXT_N, BATCH, HEADS, DIM, PAGES, MAX_LEN = 5, 2, 4, 8, 3, 64
ROWS = BATCH * NEXT_N


@pytest.fixture
def ops(monkeypatch):
    """sglang's runtime context at DCP 2 with NEXT_N draft tokens, and both logits ops faked."""
    context = types.ModuleType("sglang.srt.runtime_context")
    context.get_parallel = lambda: types.SimpleNamespace(attn_dcp_size=2)
    context.get_spec = lambda: types.SimpleNamespace(speculative_num_draft_tokens=NEXT_N)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", context)
    calls = []

    def verify(**kwargs):
        calls.append(("verify", kwargs))
        return kwargs["logits"]

    verify.fallback_reason = lambda **kwargs: None

    def rows(**kwargs):
        calls.append(("rows", kwargs))
        return kwargs["logits"]

    monkeypatch.setattr(api, "fp8_paged_mqa_logits_verify", verify)
    monkeypatch.setattr(api, "fp8_paged_mqa_logits", rows)
    monkeypatch.setattr(api, "_PAGED_TOPK_MODE", [None])
    return calls


def logits_call():
    q = torch.zeros(ROWS, HEADS, DIM, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(16, 64, dtype=torch.uint8)
    weights = torch.zeros(ROWS, HEADS)
    seq_lens = torch.full((ROWS,), MAX_LEN, dtype=torch.int32)
    tables = torch.arange(ROWS * PAGES, dtype=torch.int32).view(ROWS, PAGES)
    return api._aiter_paged_mqa_logits(
        q, kv, weights, seq_lens, tables, MAX_LEN, preshuffle=False, kv_block_size=64
    )


def paged_topk(rows, is_target_verify):
    batch = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_target_verify=lambda: is_target_verify)
    )
    q = torch.zeros(rows, HEADS, DIM, dtype=torch.float8_e4m3fn)
    return api._get_topk_paged(object(), batch, 0, q, None, None)


def test_the_paged_topk_route_notes_the_mode_and_hands_the_call_back():
    api._PAGED_TOPK_MODE[0] = None
    try:
        assert paged_topk(ROWS, True) is FallbackReason.OUT_OF_CONTRACT
        assert api._PAGED_TOPK_MODE[0] == (ROWS, True)
    finally:
        api._PAGED_TOPK_MODE[0] = None


@pytest.mark.parametrize(
    "noted",
    [None, (ROWS, False), (ROWS + NEXT_N, True)],
    ids=["unnoted", "decode", "other_rows"],
)
def test_rows_not_noted_as_this_verify_stay_on_the_row_op(ops, noted):
    api._PAGED_TOPK_MODE[0] = noted
    logits_call()
    assert [kind for kind, _ in ops] == ["rows"]
    assert api._PAGED_TOPK_MODE[0] is None


def test_a_noted_verify_is_regrouped_once(ops):
    paged_topk(ROWS, True)
    logits_call()
    ((kind, kwargs),) = ops
    assert kind == "verify"
    assert kwargs["q"].shape[:2] == (BATCH, NEXT_N)
    # each request's first row of the repeated table
    assert torch.equal(kwargs["block_tables"][:, 0], torch.tensor([0, NEXT_N * PAGES], dtype=torch.int32))
    # the note is taken: the next call, unnoted, is a row call
    logits_call()
    assert [kind for kind, _ in ops] == ["verify", "rows"]
