# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The routes on sglang's aiter calls: what they leave to aiter before AKLR is asked."""

import pytest

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import aiter_calls


@pytest.fixture
def mla_decode(monkeypatch):
    """AKLR's mla_decode_fwd, its calls recorded."""
    calls = []
    monkeypatch.setattr(
        aiter_calls.aklr_aiter, "mla_decode_fwd", lambda *args, **kwargs: calls.append(kwargs)
    )
    return calls


@pytest.mark.parametrize(
    "layout",
    [
        {"kv_layout": "head_major"},
        {"kv_csr_layout": "token"},
        {"seq_lens": object()},
        {"page_table": object()},
    ],
    ids=["kv_layout", "kv_csr_layout", "seq_lens", "page_table"],
)
def test_an_mla_decode_on_another_kv_layout_stays_on_aiter(mla_decode, layout):
    assert aiter_calls._mla_decode_fwd(*[None] * 8, **layout) is FallbackReason.OUT_OF_CONTRACT
    assert mla_decode == []


def test_an_mla_decode_on_the_token_major_page_pool_reaches_aklr(mla_decode):
    aiter_calls._mla_decode_fwd(*[None] * 8)
    assert len(mla_decode) == 1


@pytest.fixture
def fused_moe(monkeypatch):
    """AKLR's route on aiter's fused_moe, its calls recorded."""
    calls = []
    monkeypatch.setattr(aiter_calls.aklr_aiter, "fused_moe", lambda *args, **kwargs: calls.append(kwargs))
    return calls


def test_the_zero_padded_intermediate_of_fp8_activation_experts_is_computed_as_stored(fused_moe):
    aiter_calls._fused_moe(*[None] * 5, intermediate_pad=64, quant_dtype_a=object())
    assert fused_moe[0]["intermediate_pad"] == 0


def test_any_other_padded_intermediate_reaches_aklr_as_sglang_names_it(fused_moe):
    aiter_calls._fused_moe(*[None] * 5, intermediate_pad=64)
    assert fused_moe[0]["intermediate_pad"] == 64
