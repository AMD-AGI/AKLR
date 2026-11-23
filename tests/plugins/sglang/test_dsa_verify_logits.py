# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on the DSA indexer's paged top-k (CPU): it runs nothing and tells AKLR's aiter provider
whether the logits asked next are a target verify's under DCP, so the provider scores those rows by
request; a decode batch, or a verify without DCP, is noted as none."""

import sys
import types

import pytest
import torch

from aklr.plugins.sglang.layers.dsa_backend import plugin as api

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
    return parallel


def paged_topk(rows, is_target_verify, orig):
    class Observer:
        def wrap(self, target, *, versions, around=None, replace=None):
            if around is not None:
                self.paged = around

    observer = Observer()
    api.declare(observer)
    batch = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_target_verify=lambda: is_target_verify)
    )
    q = torch.zeros(rows, HEADS, DIM, dtype=torch.float8_e4m3fn)
    return observer.paged(orig, object(), batch, 0, q, marker=17)


@pytest.mark.parametrize(("verify", "dcp"), [(True, 2), (False, 2), (True, 1)])
def test_indexer_receives_its_arguments_and_returns_its_own_result(context, verify, dcp):
    context.attn_dcp_size = dcp

    def orig(indexer, batch, layer_id, q, *, marker):
        assert batch.forward_mode.is_target_verify() == verify
        return (layer_id, q.shape[0], marker)

    assert paged_topk(ROWS, verify, orig) == (0, ROWS, 17)
