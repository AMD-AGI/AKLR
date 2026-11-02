# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A process that does not serve DeepSeek-V4.1 (DeepSeek-V4 on the same classes, another
DSpark draft, another GPU or TP size) keeps sglang's code in every route around the
monokernels."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.models.deepseek_v4 as dsv4
from aklr.plugins.sglang.models.deepseek_v4 import (
    accept,
    engram_routes,
    inject,
    proposal,
    radix_routes,
    tokenize_routes,
)


@pytest.fixture
def not_serving(monkeypatch):
    monkeypatch.setattr(dsv4, "_SERVING", False)


def _sglang(*args, **kwargs):
    return "sglang's"


def test_the_verify_accept_builds_no_exchange(monkeypatch, not_serving):
    monkeypatch.setattr(accept, "_XCHG", None)
    monkeypatch.setattr(accept, "_build_xchg", lambda: pytest.fail("built"))
    assert accept._epilogue_init(_sglang, object()) == "sglang's"
    assert accept._XCHG is None


def test_the_simulated_accept_is_not_folded(not_serving):
    epilogue = SimpleNamespace(_aklr_armed=2, _static_step_state=(2, True))
    assert accept._begin_static_step(_sglang, epilogue, 2, False) == "sglang's"
    executor = SimpleNamespace(verify_epilogue=None)
    ids = torch.zeros(2, 6)
    assert accept._run_non_compact(_sglang, executor, verify_ids_2d=ids) == "sglang's"


def test_the_draft_keeps_sglangs_head_and_steps(not_serving):
    draft = SimpleNamespace(_use_fp32_lm_head=False, lm_head=object(), markov_head=None)
    assert proposal._attach(_sglang, draft) == "sglang's"
    assert "_aklr_head_fp8" not in draft.__dict__
    assert proposal._markov_block(_sglang, greedy_draft=True) == "sglang's"
    assert proposal._draft_params(_sglang, bs=1, out=None) == "sglang's"


def test_the_inject_keeps_sglangs_stores(not_serving):
    if torch.version.hip is not None:
        assert inject._why_not(object(), object()) is not None
    assert inject._inject_mla(_sglang, object(), pool=object()) == "sglang's"


def test_engram_keeps_sglangs_reduction(monkeypatch, not_serving):
    monkeypatch.setattr(engram_routes, "_custom_group", lambda v: pytest.fail("custom"))
    assert engram_routes._reduce_owned_rows(_sglang, object(), torch.zeros(1)) == ("sglang's")


def test_the_radix_insert_walk_is_sglangs(monkeypatch, not_serving):
    core = ModuleType("unified_tree_core")
    core.InsertStepResult = object
    core._InsertPhase = SimpleNamespace(WALK="walk")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.unified_cache.unified_tree_core", core)
    tree = SimpleNamespace(_ongoing_insert_walk_state=SimpleNamespace(phase="walk"))
    assert radix_routes._advance_insert(_sglang, tree) == "sglang's"


class _Tokenizer:
    is_fast = True

    def __init__(self):
        self.added_tokens_decoder = {}

    def encode(self, text, **kwargs):
        return [len(text)]


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
def test_the_token_cache_is_deepseek_v41s(model_type):
    """The tokenizer's process has no GPU: the config alone decides."""
    manager = SimpleNamespace(
        tokenizer=_Tokenizer(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )
    plain = manager.tokenizer.encode
    tokenize_routes._init(lambda self: None, manager)
    cached = isinstance(manager.tokenizer.encode, tokenize_routes._SegmentEncode)
    assert cached == (model_type == "deepseek_v41")
    assert cached or manager.tokenizer.encode == plain


def test_the_dp_pad_mask_is_sglangs(not_serving):
    from aklr.plugins.sglang.models.deepseek_v4 import dp_routes

    calls = []
    batch = SimpleNamespace(num_token_non_padded=torch.tensor(1))
    dp_routes._mask_dp_pad_rows(lambda x, b: calls.append("sglang's"), torch.ones(2, 4), batch)
    assert calls == ["sglang's"]
