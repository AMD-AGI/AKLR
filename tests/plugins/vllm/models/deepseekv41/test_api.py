# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models import deepseekv41
from aklr.plugins.vllm.models.deepseekv41 import api

_HEADS, _TOPK = 8, 8
_OPS = ("deepseekv41_sparse_mla",)


def pool(pages, page_size, row_bytes=None):
    """A vLLM fp8_ds_mla cache: a page of records in each block's row, which other caches' pages share."""
    row_bytes = row_bytes or 3 * page_size * 584
    return torch.zeros(pages, row_bytes, dtype=torch.uint8).as_strided(
        (pages, page_size, 584), (row_bytes, 584, 1)
    )


@pytest.fixture
def asked(monkeypatch):
    """The calls the routes ask the ops for, by op; every op answers NO_WORKLOAD."""
    calls = {}
    for name in _OPS:
        op = SimpleNamespace(
            fallback_reason=lambda _name=name, **call: (
                calls.setdefault(_name, []).append(call) or FallbackReason.NO_WORKLOAD
            )
        )
        monkeypatch.setattr(api, name, op)
    return calls


@pytest.fixture
def step(monkeypatch):
    """A decode-only step of 64 rows in vLLM's forward context; change it to make another."""
    swa = SimpleNamespace(
        num_prefill_tokens=0,
        block_size=32,
        decode_swa_width=128,
        decode_swa_ragged_indices=torch.zeros(64 * 128, dtype=torch.int32),
        prefill_swa_ragged_indices=None,
        prefill_swa_indices=None,
    )
    context = SimpleNamespace(attn_metadata={"swa": swa, "compressed": SimpleNamespace(block_size=128)})
    forward_context = ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context)
    return swa


def attention(**changes):
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=pool(3, 32)),
        compressed_cache_prefix="compressed",
        compress_ratio=2,
        _compressed_kv_cache=lambda: pool(5, 64),
        topk_indices_buffer=torch.zeros(1024, _TOPK, dtype=torch.int32),
        attn_sink=torch.zeros(_HEADS),
        scale=512**-0.5,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(16, 64)),
    )
    attn.__dict__.update(changes)
    return attn


def sparse_mla(attn, swa, tokens=64, **changes):
    arguments = {
        "q": torch.zeros(tokens, _HEADS, 512, dtype=torch.bfloat16),
        "output": None,
        "swa_k_cache": attn.swa_cache_layer.kv_cache,
        "swa_indices": swa.decode_swa_ragged_indices,
        "swa_indptr": torch.zeros(tokens + 1, dtype=torch.int32),
        "compressed_k_cache": attn._compressed_kv_cache(),
        "topk_indices": torch.zeros(tokens * _TOPK, dtype=torch.int32),
        "topk_indptr": torch.zeros(tokens + 1, dtype=torch.int32),
        "positions": torch.zeros(tokens, dtype=torch.int64),
        "output_mxfp8": (torch.zeros(tokens, _HEADS * 512), torch.zeros(tokens, _HEADS * 16)),
    }
    return api._aiter_sparse_mla(attn, **{**arguments, **changes})


def test_a_decode_steps_sparse_mla_is_asked_for_at_vllms_pools_and_index_widths(asked, step):
    """vLLM hands its kernel ragged index lists and stores MXFP8: the op is asked for at the
    widths the lists were packed from, and the step stays on vLLM."""
    assert sparse_mla(attention(), step) is FallbackReason.NO_WORKLOAD
    (call,) = asked["deepseekv41_sparse_mla"]
    shapes = {name: tuple(value.shape) for name, value in call.items() if isinstance(value, torch.Tensor)}
    assert shapes == {
        "q": (64, 1, _HEADS, 512),
        "kv_cache": (3, 32, 1, 584),
        "indices": (64, 1, 128),
        "sinks": (_HEADS,),
        "extra_kv_cache": (5, 64, 1, 584),
        "extra_indices": (64, 1, _TOPK),
        "rope_freqs": (16, 64),
        "positions": (64,),
    }
    assert call["kv_cache_layout"] == "strided_pages"
    assert call["kv_cache"].stride() == (3 * 32 * 584, 584, 584, 1)
    assert call["extra_kv_cache"].stride() == (3 * 64 * 584, 584, 584, 1)


def test_a_steps_positions_reach_sparse_mla_on_the_16_bytes_dispatch_requires(asked, step):
    # vLLM passes the positions past a first row: int64, so 8 bytes past a boundary
    sparse_mla(attention(), step, positions=torch.arange(65)[1:])
    (call,) = asked["deepseekv41_sparse_mla"]
    assert call["positions"].tolist() == list(range(1, 65)) and call["positions"].data_ptr() % 16 == 0


def test_a_prefills_sparse_mla_is_asked_for_at_the_prefill_index_width(asked, step):
    step.num_prefill_tokens = 16
    step.prefill_swa_ragged_indices = torch.zeros(7, dtype=torch.int32)
    step.prefill_swa_indices = torch.zeros(16, 96, dtype=torch.int32)
    reason = sparse_mla(
        attention(),
        step,
        tokens=16,
        swa_indices=step.prefill_swa_ragged_indices,
        compressed_k_cache=None,
        topk_indices=None,
        topk_indptr=None,
    )
    assert reason is FallbackReason.NO_WORKLOAD
    (call,) = asked["deepseekv41_sparse_mla"]
    assert call["indices"].shape == (16, 1, 96)
    assert "extra_kv_cache" not in call


@pytest.mark.parametrize(
    "changes",
    [
        {"swa_k_cache": torch.zeros(3, 32, 528, dtype=torch.uint8)},
        {"swa_indices": torch.zeros(4, dtype=torch.int32)},
    ],
    ids=["records_of_another_width", "indices_of_neither_phase"],
)
def test_a_sparse_mla_the_contract_cannot_express_is_not_asked_for(asked, step, changes):
    assert sparse_mla(attention(), step, **changes) is FallbackReason.OUT_OF_CONTRACT
    assert asked == {}


def shapes(call):
    return {name: tuple(value.shape) for name, value in call.items() if isinstance(value, torch.Tensor)}


def point(name, call):
    """The Point the op's contract keys ``call`` at, as dispatch would; None outside the contract."""
    codec = FastKeyCodec.from_contract(ContractRegistry.default().get(f"{name}@1.0"))
    bound = inspect.signature(getattr(deepseekv41, name)).bind(**call)
    bound.apply_defaults()
    key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()(dict(bound.arguments))
    return None if key is None else codec.to_point(key)


def mxfp8_layer(scale_rows=2):
    return SimpleNamespace(
        weight=torch.zeros(64, 128, dtype=torch.float8_e4m3fn),
        weight_scale=torch.zeros(scale_rows, 4, dtype=torch.uint8),
    )


def test_a_block_scaled_mxfp8_linear_is_one_op_call_on_the_rows_then_vllms_bias(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api,
        "deepseekv41_mxfp8_linear",
        lambda **call: calls.append(call) or torch.ones(6, 64, dtype=torch.bfloat16),
    )
    bias = torch.full((64,), 2.0, dtype=torch.bfloat16)
    out = api._apply_weights(None, mxfp8_layer(), torch.zeros(2, 3, 128, dtype=torch.bfloat16), bias)
    assert out.shape == (2, 3, 64) and bool((out == 3).all())
    assert calls[0]["x"].shape == (6, 128) and "x_scale" not in calls[0]


def test_an_activation_vllm_holds_quantized_goes_with_its_scales(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api,
        "deepseekv41_mxfp8_linear",
        lambda **call: calls.append(call) or torch.ones(6, 64, dtype=torch.bfloat16),
    )
    quantized = SimpleNamespace(
        data=torch.zeros(6, 128, dtype=torch.float8_e4m3fn),
        scale=torch.zeros(6, 4, dtype=torch.uint8),
        orig_dtype=torch.bfloat16,
        orig_shape=(6, 128),
    )
    assert api._apply_weights(None, mxfp8_layer(), quantized).shape == (6, 64)
    assert calls[0]["x_scale"].shape == (6, 4)


@pytest.mark.parametrize(
    ("layer", "x"),
    [
        (mxfp8_layer(scale_rows=64), torch.zeros(4, 128, dtype=torch.bfloat16)),
        (mxfp8_layer(), torch.zeros(4, 128, dtype=torch.float16)),
        (mxfp8_layer(), torch.zeros(0, 128, dtype=torch.bfloat16)),
    ],
    ids=["a_scale_per_weight_row", "an_fp16_input", "no_rows"],
)
def test_an_mxfp8_linear_outside_the_contract_stays_on_vllm(monkeypatch, layer, x):
    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", lambda **call: pytest.fail("asked"))
    assert api._apply_weights(None, layer, x) is FallbackReason.OUT_OF_CONTRACT


def wo_a_bmm(groups, tokens=6, scale_rows=None, op=None):
    wo_a = SimpleNamespace(
        weight=torch.zeros(groups * 64, 4096, dtype=torch.float8_e4m3fn),
        weight_scale=torch.zeros(scale_rows or groups * 2, 128, dtype=torch.uint8),
    )
    a = torch.zeros(tokens, groups * 4096, dtype=torch.float8_e4m3fn)
    return api._mxfp8_wo_a_bmm(a, torch.zeros(tokens, groups * 128, dtype=torch.uint8), wo_a, groups, 64)


def test_a_grouped_mxfp8_wo_a_is_one_linear_a_group_on_the_groups_own_rows(monkeypatch):
    calls = []

    def linear(**call):
        calls.append(call)
        return torch.full((6, 64), float(len(calls)), dtype=torch.bfloat16)

    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", linear)
    out = wo_a_bmm(groups=2)
    # group-major columns, as vLLM's kernel writes them
    assert out.shape == (6, 128) and out[0, ::64].tolist() == [1.0, 2.0]
    assert [shapes(call) for call in calls] == [
        {"x": (6, 4096), "x_scale": (6, 128), "weight": (64, 4096), "weight_scale": (2, 128)}
    ] * 2
    assert all(point("deepseekv41_mxfp8_linear", call).axes == {"m": 6, "n": 64, "k": 4096} for call in calls)


@pytest.mark.parametrize(
    "changes", [{"scale_rows": 128}, {"tokens": 0}], ids=["a_scale_per_weight_row", "no_rows"]
)
def test_a_wo_a_outside_the_linears_contract_stays_on_vllm(monkeypatch, changes):
    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", lambda **call: pytest.fail("asked"))
    assert wo_a_bmm(groups=2, **changes) is FallbackReason.OUT_OF_CONTRACT


def test_a_wo_a_group_no_kernel_serves_leaves_the_whole_product_to_vllm(monkeypatch):
    monkeypatch.setattr(api, "deepseekv41_mxfp8_linear", lambda **call: FallbackReason.NO_WORKLOAD)
    assert wo_a_bmm(groups=2) is FallbackReason.NO_WORKLOAD


def compressor(hidden=4096, head=512):
    """A layer's compressor: the fused KV/score weight, [2 x head_dim, hidden] in bf16."""
    return SimpleNamespace(
        fused_wkv_wgate=SimpleNamespace(weight=torch.zeros(2 * head, hidden, dtype=torch.bfloat16))
    )


def test_the_compressors_kv_score_projection_is_one_gemm_with_fp32_out(monkeypatch):
    calls, kv_score = [], torch.ones(6, 1024)
    monkeypatch.setattr(api, "gemm", lambda a, b, **call: calls.append((a, b, call)) or kv_score)
    linear, x = compressor().fused_wkv_wgate, torch.zeros(6, 4096, dtype=torch.bfloat16)
    assert api._kv_score_forward(linear, x) is kv_score
    ((a, b, call),) = calls
    assert a is x and b is linear.weight and call == {"out_dtype": torch.float32}


def test_a_kv_score_projection_no_kernel_serves_stays_on_vllm(monkeypatch):
    monkeypatch.setattr(api, "gemm", lambda a, b, **call: FallbackReason.NO_WORKLOAD)
    x = torch.zeros(6, 4096, dtype=torch.bfloat16)
    assert api._kv_score_forward(compressor().fused_wkv_wgate, x) is FallbackReason.NO_WORKLOAD


@pytest.fixture
def gemm_asked(monkeypatch):
    """The gemms a route asks for without running them; the op answers NO_WORKLOAD."""
    calls = []
    op = SimpleNamespace(
        fallback_reason=lambda a, b, **call: calls.append((a, b, call)) or FallbackReason.NO_WORKLOAD
    )
    monkeypatch.setattr(api, "gemm", op)
    return calls


def test_the_projection_inside_the_multi_stream_chain_is_asked_for_and_left_to_vllm(gemm_asked):
    """vLLM calls torch.mm inline there, between its stream forks: the route records the call."""
    attn, x = SimpleNamespace(compressor=compressor()), torch.zeros(6, 4096, dtype=torch.bfloat16)
    assert api._forward_csa2_full(attn, x, None) is FallbackReason.NO_WORKLOAD
    ((a, b, call),) = gemm_asked
    assert a is x and b is attn.compressor.fused_wkv_wgate.weight and call == {"out_dtype": torch.float32}


def test_the_projection_inside_a_decode_steps_selection_is_asked_for_and_left_to_vllm(gemm_asked):
    layer, x = (
        SimpleNamespace(attn=SimpleNamespace(compressor=compressor())),
        torch.zeros(6, 4096, dtype=torch.bfloat16),
    )
    assert api._select(layer, None, x, None) is FallbackReason.NO_WORKLOAD
    ((a, b, call),) = gemm_asked
    assert (
        a is x and b is layer.attn.compressor.fused_wkv_wgate.weight and call == {"out_dtype": torch.float32}
    )


def test_a_selection_on_a_layer_without_a_compressor_asks_for_nothing(gemm_asked):
    layer = SimpleNamespace(attn=SimpleNamespace(compressor=None))
    assert api._select(layer, None, torch.zeros(6, 4096), None) is FallbackReason.OUT_OF_CONTRACT
    assert gemm_asked == []
