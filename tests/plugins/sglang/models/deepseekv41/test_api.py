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


def wo_a_call(tokens=3, groups=2, block=32):
    """sglang's wo_a projection of a rank as its BMM takes it: each group's quantized rows, its weight in aiter's
    16 x 16 tiles and the compact block scales."""
    return (
        torch.zeros(tokens, groups, 128, dtype=torch.float8_e4m3fn),
        torch.zeros(groups, 64, 128, dtype=torch.float8_e4m3fn),
        torch.arange(tokens * groups * 4, dtype=torch.uint8).view(tokens, groups, 4),
        torch.zeros(groups, 64 // block, 128 // block, dtype=torch.uint8),
    )


def test_wo_a_reaches_the_mxfp8_linear_once_a_group_on_its_shuffled_weight(monkeypatch):
    asked = []

    def op(**call):
        asked.append(call)
        return torch.full((3, 64), float(len(asked)), dtype=torch.bfloat16)

    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", op)
    o_fp8, weight, o_scale, weight_scale = wo_a_call()
    out = api._wo_a_bmm(o_fp8, weight, o_scale, weight_scale)
    assert len(asked) == 2
    for group, call in enumerate(asked):
        assert set(call) == {"x", "x_scale", "weight", "weight_scale", "weight_layout"}
        assert call["weight_layout"] == "shuffled_16x16"
        # the group's rows, row-major on the 16 bytes dispatch requires
        assert call["x"].shape == (3, 128) and call["x"].is_contiguous() and call["x"].data_ptr() % 16 == 0
        assert call["x_scale"].is_contiguous() and call["x_scale"].data_ptr() % 16 == 0
        assert call["x_scale"].tolist() == [[4 * (2 * t + group) + i for i in range(4)] for t in range(3)]
        assert call["weight"].data_ptr() == weight[group].data_ptr() and call["weight"].shape == (64, 128)
        assert call["weight_scale"].data_ptr() == weight_scale[group].data_ptr()
    # as aiter's BMM returns it: a token's groups side by side
    assert out.shape == (3, 2, 64) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out[:, 0].eq(1).all() and out[:, 1].eq(2).all()


@pytest.mark.parametrize(
    "call",
    [wo_a_call(block=64), wo_a_call(tokens=0), tuple(t.flatten(0, 1) for t in wo_a_call())],
    ids=["other_block", "no_token", "ungrouped"],
)
def test_a_wo_a_the_op_cannot_express_stays_on_sglang(monkeypatch, call):
    asked = []
    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", lambda **call: asked.append(call))
    assert api._wo_a_bmm(*call) is FallbackReason.OUT_OF_CONTRACT and asked == []


def test_a_wo_a_group_the_op_declines_stays_on_sglang(monkeypatch):
    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", lambda **call: FallbackReason.NO_WORKLOAD)
    assert api._wo_a_bmm(*wo_a_call()) is FallbackReason.NO_WORKLOAD


class _Gemm:
    """AKLR's gemm as the routes see it: what each is asked and what each runs, the op declining with ``reason``
    and answering a run with the run's number at the op's output shape."""

    def __init__(self):
        self.asked, self.ran, self.reason = [], [], None

    def fallback_reason(self, a, b, **kwargs):
        self.asked.append(dict(a=a, b=b, **kwargs))
        return self.reason

    def __call__(self, a, b):
        self.ran.append((a, b))
        return torch.full((a.shape[0], b.shape[0]), float(len(self.ran)), dtype=a.dtype)


@pytest.fixture
def gemm(monkeypatch):
    fake = _Gemm()
    monkeypatch.setattr(api, "gemm", fake)
    return fake


def bf16_wo_a_call(tokens=3, groups=2):
    """sglang's bf16 wo_a projection of a rank: each token's groups side by side, and a weight a group."""
    o = torch.arange(tokens * groups * 128, dtype=torch.bfloat16).view(tokens, groups, 128)
    return o, torch.zeros(groups, 64, 128, dtype=torch.bfloat16)


def test_a_prefills_bf16_wo_a_reaches_gemm_once_a_group(gemm):
    o, wo_a = bf16_wo_a_call()
    out = api._apply_wo_a_bf16_matmul(o, wo_a, is_decode=False, is_prefill=True, fast_path=True)
    assert len(gemm.asked) == len(gemm.ran) == 2
    for group, (asked, (a, b)) in enumerate(zip(gemm.asked, gemm.ran, strict=True)):
        # asked with the shape and dtype of a group's rows, on the 16 bytes dispatch requires
        assert set(asked) == {"a", "b"} and asked["a"].shape == (3, 128) and asked["a"].dtype == o.dtype
        assert asked["a"].is_contiguous() and asked["a"].data_ptr() % 16 == 0
        # run on the group's rows, row-major, and its weight as sglang holds it
        assert a.is_contiguous() and a.data_ptr() % 16 == 0 and a.equal(o[:, group])
        assert asked["b"].data_ptr() == b.data_ptr() == wo_a[group].data_ptr() and b.shape == (64, 128)
    # as sglang's einsum returns it: a token's groups side by side
    assert out.shape == (3, 2, 64) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out[:, 0].eq(1).all() and out[:, 1].eq(2).all()


@pytest.mark.parametrize(
    ("call", "mode"),
    [
        (tuple(t.flatten(0, 1) for t in bf16_wo_a_call()), {"is_decode": False}),
        ((bf16_wo_a_call()[0], bf16_wo_a_call(groups=1)[1]), {"is_decode": False}),
    ],
    ids=["ungrouped", "other_groups"],
)
def test_a_bf16_wo_a_sglangs_own_kernels_take_stays_on_sglang(gemm, call, mode):
    assert api._apply_wo_a_bf16_matmul(*call, **mode) is FallbackReason.OUT_OF_CONTRACT
    assert gemm.asked == gemm.ran == []


def test_a_bf16_wo_a_gemm_declines_stays_on_sglang_uncopied(gemm):
    gemm.reason = FallbackReason.NO_WORKLOAD
    assert api._apply_wo_a_bf16_matmul(*bf16_wo_a_call(), is_decode=False) is FallbackReason.NO_WORKLOAD
    assert len(gemm.asked) == 1 and gemm.ran == []


@pytest.mark.parametrize(
    "mode",
    [{"is_decode": True}, {"is_decode": False, "is_target_verify": True}],
    ids=["decode", "target_verify"],
)
def test_a_decode_or_verify_bf16_wo_a_is_asked_of_gemm_once_a_group_and_left_to_sglang(gemm, mode):
    o, wo_a = bf16_wo_a_call()
    # sglang's kernels return these rows in other forms: the calls are recorded, never run
    assert api._apply_wo_a_bf16_matmul(o, wo_a, **mode) is FallbackReason.OUT_OF_CONTRACT
    assert [(asked["a"].shape, asked["b"].data_ptr()) for asked in gemm.asked] == [
        ((3, 128), wo_a[group].data_ptr()) for group in range(2)
    ]
    assert gemm.ran == []


@pytest.mark.parametrize(
    "answer", [torch.ones(4, 256), FallbackReason.NO_WORKLOAD], ids=["a_kernel_serves", "the_op_declines"]
)
def test_a_bf16_linear_with_fp32_output_reaches_gemm(monkeypatch, answer):
    asked = []
    monkeypatch.setattr(api, "gemm", lambda *args, **kwargs: asked.append((args, kwargs)) or answer)
    x, weight = torch.zeros(4, 512, dtype=torch.bfloat16), torch.zeros(256, 512, dtype=torch.bfloat16)
    assert api._linear_bf16_fp32(x, weight, hpc_kernel_min_m=8) is answer
    (((a, b), kwargs),) = asked
    assert a is x and b is weight and kwargs == {"out_dtype": torch.float32}


@pytest.mark.parametrize(
    "reason", [None, FallbackReason.NO_WORKLOAD], ids=["a_kernel_serves", "the_op_declines"]
)
def test_the_indexers_key_projection_is_asked_of_gemm_and_left_to_sglang(gemm, reason):
    gemm.reason = reason
    indexer = SimpleNamespace(wk=SimpleNamespace(weight=torch.zeros(128, 512, dtype=torch.bfloat16)))
    latent = torch.zeros(4, 512, dtype=torch.bfloat16)
    # sglang's small-row kernel keeps its own reduction order: the call is recorded, never run
    assert api._forward_wk(indexer, latent) is (reason or FallbackReason.OUT_OF_CONTRACT)
    (asked,) = gemm.asked
    assert set(asked) == {"a", "b"} and asked["a"] is latent and asked["b"] is indexer.wk.weight
    assert gemm.ran == []


@pytest.mark.parametrize(("ratio", "asks"), [(1, 1), (4, 0)], ids=["uncompressed", "compressed"])
def test_an_uncompressed_layers_kv_projection_is_asked_of_gemm_and_left_to_sglang(gemm, ratio, asks):
    compressor = SimpleNamespace(
        compress_ratio=ratio, wkv=SimpleNamespace(weight=torch.zeros(512, 4096, dtype=torch.bfloat16))
    )
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    # a compressed layer's projections reach gemm through sglang's bf16 linear with fp32 output
    assert api._project(compressor, x) is FallbackReason.OUT_OF_CONTRACT
    assert [(asked["a"], asked["b"]) for asked in gemm.asked] == [(x, compressor.wkv.weight)] * asks
    assert gemm.ran == []


@pytest.mark.parametrize(
    "reason", [None, FallbackReason.NO_WORKLOAD], ids=["a_kernel_serves", "the_op_declines"]
)
def test_the_routers_gemv_is_asked_of_gemm_as_fp32_logits_and_left_to_sglang(gemm, reason):
    gemm.reason = reason
    hidden, weight = torch.zeros(4, 512, dtype=torch.bfloat16), torch.zeros(384, 512, dtype=torch.bfloat16)
    # sglang's split-K partials are not the op's output: the call is recorded, never run
    assert api._rocm_router_gemv_split_k(hidden, weight) is (reason or FallbackReason.OUT_OF_CONTRACT)
    (asked,) = gemm.asked
    assert set(asked) == {"a", "b", "out_dtype"} and asked["out_dtype"] == torch.float32
    assert asked["a"] is hidden and asked["b"] is weight and gemm.ran == []
