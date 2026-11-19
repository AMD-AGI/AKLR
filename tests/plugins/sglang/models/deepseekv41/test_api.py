# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's DeepSeek-V4.1 routes hand the ops, on tensors shaped as sglang's."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseekv41 import api


def test_a_steps_positions_reach_sparse_mla_on_the_16_bytes_dispatch_requires(monkeypatch):
    asked = []
    monkeypatch.setattr(
        api, "deepseekv41_sparse_mla", lambda **call: asked.append(call) or FallbackReason.NO_WORKLOAD
    )
    # sglang passes a step's positions as a slice of the batch's: int64, so 8 bytes past a boundary at an odd row
    positions = torch.arange(5)[1:]
    reason = api._aiter_sparse_decode_fwd(
        torch.zeros(4, 1, 16, 512, dtype=torch.bfloat16),
        torch.zeros(2, 64, 1, 584, dtype=torch.uint8),
        torch.zeros(4, 1, 64, dtype=torch.int32),
        torch.zeros(16),
        0.044,
        inv_rope=(torch.zeros(16, 64), positions),
    )
    assert reason is FallbackReason.NO_WORKLOAD
    (call,) = asked
    assert call["positions"].tolist() == [1, 2, 3, 4] and call["positions"].data_ptr() % 16 == 0


@pytest.fixture
def mxfp8_linear(monkeypatch):
    """The MXFP8 linear op: its calls recorded, each answered with ones of the op's output shape."""
    asked = []

    def op(**call):
        asked.append(call)
        return torch.ones(call["x"].shape[0], call["weight"].shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", op)
    return asked


def mxfp8_layer():
    """A layer of sglang's aiter route as its long prefills read it: the plain weight, its compact block scales."""
    return SimpleNamespace(
        weight=torch.zeros(64, 128, dtype=torch.float8_e4m3fn),
        weight_scale_mx_e8m0=torch.zeros(2, 4, dtype=torch.uint8),
    )


def test_a_long_prefills_mxfp8_linear_reaches_the_op_on_the_layers_plain_weight(mxfp8_linear):
    layer, x = mxfp8_layer(), torch.zeros(3, 5, 128, dtype=torch.bfloat16)
    out = api._apply_flydsl_mxfp8(layer, x, torch.full((64,), 2.0, dtype=torch.bfloat16), None)
    (call,) = mxfp8_linear
    assert set(call) == {"x", "weight", "weight_scale"}
    assert call["x"].shape == (15, 128) and call["x"].data_ptr() == x.data_ptr()
    assert call["weight"] is layer.weight and call["weight_scale"] is layer.weight_scale_mx_e8m0
    # the op's ones, then sglang's bias, at x's leading shape
    assert out.shape == (3, 5, 64) and out.dtype == torch.bfloat16 and (out == 3).all()


def test_a_long_prefills_quantized_input_reaches_the_op_with_its_scales(mxfp8_linear):
    scale = torch.zeros(15, 4, dtype=torch.uint8)
    out = api._apply_flydsl_mxfp8(mxfp8_layer(), torch.zeros(15, 128, dtype=torch.float8_e4m3fn), None, scale)
    (call,) = mxfp8_linear
    assert call["x_scale"].shape == (15, 4) and call["x_scale"].data_ptr() == scale.data_ptr()
    assert out.shape == (15, 64) and (out == 1).all()


def test_a_long_prefills_unquantized_input_sglang_casts_stays_on_sglang(mxfp8_linear):
    reason = api._apply_flydsl_mxfp8(mxfp8_layer(), torch.zeros(15, 128), None, None)
    assert reason is FallbackReason.OUT_OF_CONTRACT and mxfp8_linear == []


class _Pool:
    """sglang's KV pool of a layer, as the attention backend reads it: a row a page, padded past its slots."""

    swa_page_size = 2

    def __init__(self, slot_bytes=584):
        self.swa_kv_pool = SimpleNamespace(kv_cache_total_dim=slot_bytes)
        self.swa = torch.arange(3 * (2 * slot_bytes + 16), dtype=torch.uint8).view(3, -1)
        self.extra = torch.zeros(5, 64 * slot_bytes + 16, dtype=torch.uint8)
        self.slot_bytes = slot_bytes

    def get_swa_key_buffer_radix(self, layer_id):
        return self.swa

    def get_extra_key_buffer(self, layer_id):
        return self.extra

    def get_extra_key_page_size(self, layer_id):
        return 64

    def get_extra_key_bytes_per_token(self, layer_id):
        return self.slot_bytes


def prefill_metadata():
    """A forward's attention metadata, padded to six rows: each row's SWA slots and top-k slots, valid up to its
    length."""
    return SimpleNamespace(
        swa_page_indices=torch.arange(6 * 64, dtype=torch.int32).view(6, 64),
        swa_topk_lengths=torch.tensor([1, 2, 3, 64, 9, 9], dtype=torch.int32),
        sparse_page_indices=lambda ratio: torch.full((6, 64), 7 * ratio, dtype=torch.int32),
        sparse_topk_lengths=lambda ratio: torch.tensor([1, 1, 2, 2, 9, 9], dtype=torch.int32),
    )


@pytest.fixture
def sparse_mla(monkeypatch):
    """The sparse MLA op, its calls recorded and answered with ones; sglang's folding of the lengths into the
    indices, which the route asks of sglang's module."""
    backend = ModuleType("sglang.srt.layers.attention.deepseek_v4_backend_hip_radix")

    def fold(metadata, ratio, indices, lengths, extra_indices, extra_lengths):
        columns = torch.arange(indices.shape[-1])
        folded = indices.masked_fill(columns >= lengths.view(-1, 1, 1), -1)
        if extra_indices is None:
            return folded, None
        return folded, extra_indices.masked_fill(columns >= extra_lengths.view(-1, 1, 1), -1)

    backend._fold_lengths_for_aiter_sparse = fold
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    asked = []

    def op(**call):
        asked.append(call)
        return torch.ones_like(call["q"])

    monkeypatch.setattr(api, "deepseekv41_sparse_mla", op)
    return asked


def test_a_long_prefills_attention_reaches_sparse_mla_on_the_slots_of_its_window(sparse_mla):
    pool, sinks = _Pool(), torch.zeros(16)
    q = torch.zeros(4, 16, 512, dtype=torch.bfloat16)
    out = api._forward_prefill_opus(
        SimpleNamespace(token_to_kv_pool=pool, softmax_scale=0.044),
        q,
        3,
        0,
        None,
        prefill_metadata(),
        sinks,
        None,
    )
    (call,) = sparse_mla
    assert set(call) == {"q", "kv_cache", "indices", "sinks", "sm_scale", "kv_cache_layout"}
    assert call["q"].shape == (4, 1, 16, 512) and call["q"].data_ptr() == q.data_ptr()
    # each page's two slots, without the row's padding
    assert call["kv_cache"].shape == (3, 2, 1, 584)
    assert (call["kv_cache"][1].flatten() == pool.swa[1, :1168]).all()
    assert call["kv_cache_layout"] == "padded_pages"
    assert call["sinks"] is sinks and call["sm_scale"] == 0.044
    # the chunk's four tokens of the six rows, each row's slots past its length skipped
    assert call["indices"].shape == (4, 1, 64)
    assert call["indices"][2, 0, :4].tolist() == [128, 129, 130, -1]
    assert (call["indices"] >= 0).sum(-1).flatten().tolist() == [1, 2, 3, 64]
    assert out.shape == (4, 16, 512) and (out == 1).all()


def test_a_long_prefills_top_k_reaches_sparse_mla_as_the_extra_pools_slots(sparse_mla):
    positions = torch.arange(4)
    api._forward_prefill_opus(
        SimpleNamespace(token_to_kv_pool=_Pool(), softmax_scale=0.044),
        torch.zeros(4, 16, 512, dtype=torch.bfloat16),
        3,
        2,
        None,
        prefill_metadata(),
        torch.zeros(16),
        (torch.zeros(16, 64), positions),
    )
    (call,) = sparse_mla
    assert call["extra_kv_cache"].shape == (5, 64, 1, 584)
    assert call["extra_indices"].shape == (4, 1, 64)
    assert call["extra_indices"][:, 0, :3].tolist() == [
        [14, -1, -1],
        [14, -1, -1],
        [14, 14, -1],
        [14, 14, -1],
    ]
    assert call["rope_freqs"].shape == (16, 64) and call["positions"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("pool", "tokens"),
    [(_Pool(slot_bytes=160), 4), (_Pool(), 7)],
    ids=["compact_slots", "more_tokens_than_rows"],
)
def test_a_long_prefill_the_op_cannot_express_stays_on_sglang(sparse_mla, pool, tokens):
    reason = api._forward_prefill_opus(
        SimpleNamespace(token_to_kv_pool=pool, softmax_scale=0.044),
        torch.zeros(tokens, 16, 512, dtype=torch.bfloat16),
        3,
        2,
        None,
        prefill_metadata(),
        torch.zeros(16),
        None,
    )
    assert reason is FallbackReason.OUT_OF_CONTRACT and sparse_mla == []


def test_a_long_prefill_the_op_declines_stays_on_sglang(sparse_mla, monkeypatch):
    monkeypatch.setattr(api, "deepseekv41_sparse_mla", lambda **call: FallbackReason.NO_WORKLOAD)
    reason = api._forward_prefill_opus(
        SimpleNamespace(token_to_kv_pool=_Pool(), softmax_scale=0.044),
        torch.zeros(4, 16, 512, dtype=torch.bfloat16),
        3,
        0,
        None,
        prefill_metadata(),
        torch.zeros(16),
        None,
    )
    assert reason is FallbackReason.NO_WORKLOAD
