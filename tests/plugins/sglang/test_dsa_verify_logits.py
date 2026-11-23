# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on the DSA indexer's paged top-k (CPU): it runs nothing and tells AKLR's aiter provider
whether the logits asked next are a target verify's under DCP, so the provider scores those rows by
request; a decode batch, or a verify without DCP, is noted as none."""

import importlib
import sys
import types

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.dsa_backend import api

mla = importlib.import_module("aklr.plugins.aiter.provider.mla")

NEXT_N, BATCH, HEADS, DIM = 5, 2, 4, 8
ROWS = BATCH * NEXT_N


@pytest.fixture
def context(monkeypatch):
    """sglang's runtime context with NEXT_N draft tokens, at the DCP size the test sets; a stale note."""
    parallel = types.SimpleNamespace(attn_dcp_size=2)
    module = types.ModuleType("sglang.srt.runtime_context")
    module.get_parallel = lambda: parallel
    module.get_spec = lambda: types.SimpleNamespace(speculative_num_draft_tokens=NEXT_N)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", module)
    monkeypatch.setattr(mla, "_TARGET_VERIFY", [(1, 1)])
    return parallel


def paged_topk(rows, is_target_verify):
    batch = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_target_verify=lambda: is_target_verify)
    )
    q = torch.zeros(rows, HEADS, DIM, dtype=torch.float8_e4m3fn)
    return api._get_topk_paged(object(), batch, 0, q, None, None)


def test_a_verify_under_dcp_is_noted_for_the_provider_and_sglang_runs_its_top_k(context):
    assert paged_topk(ROWS, True) is FallbackReason.OUT_OF_CONTRACT
    assert mla._TARGET_VERIFY[0] == (ROWS, NEXT_N)


def test_a_decode_batch_clears_the_note(context):
    assert paged_topk(ROWS, False) is FallbackReason.OUT_OF_CONTRACT
    assert mla._TARGET_VERIFY[0] is None


def test_a_verify_without_dcp_clears_the_note(context):
    context.attn_dcp_size = 1
    paged_topk(ROWS, True)
    assert mla._TARGET_VERIFY[0] is None
