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

# one group of 8 heads on the rank, whose queries fit in the four streams
_HIDDEN, _Q_RANK, _HEADS, _TOPK, _O_RANK, _EXPERTS, _INTER = 1024, 32, 8, 8, 64, 16, 128
_OPS = (
    "deepseekv41_sparse_mla",
    "deepseekv41_mono_k1",
    "deepseekv41_mono_k2_attention",
    "deepseekv41_mono_k2_ffn",
)


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
        num_prefills=0,
        num_decodes=64,
        num_decode_tokens=64,
        num_prefill_tokens=0,
        block_size=32,
        decode_swa_width=128,
        decode_swa_indices=torch.zeros(64, 1, 128, dtype=torch.int32),
        decode_swa_ragged_indices=torch.zeros(64 * 128, dtype=torch.int32),
        prefill_swa_ragged_indices=None,
        prefill_swa_indices=None,
    )
    context = SimpleNamespace(attn_metadata={"swa": swa, "compressed": SimpleNamespace(block_size=128)})
    forward_context = ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: context
    forward_context.is_forward_context_available = lambda: True
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context)
    return swa


def attention(**changes):
    def linear(rows, columns):
        return SimpleNamespace(
            weight=torch.zeros(rows, columns, dtype=torch.float8_e4m3fn),
            weight_scale=torch.zeros(rows // 32, columns // 32, dtype=torch.uint8),
        )

    def norm(width):
        return SimpleNamespace(weight=torch.zeros(width, dtype=torch.bfloat16), variance_epsilon=1e-6)

    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=pool(3, 32)),
        compressed_cache_prefix="compressed",
        compress_ratio=2,
        _compressed_kv_cache=lambda: pool(5, 64),
        topk_indices_buffer=torch.zeros(1024, _TOPK, dtype=torch.int32),
        n_local_heads=_HEADS,
        attn_sink=torch.zeros(_HEADS),
        scale=512**-0.5,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(16, 64)),
        fused_wqa_wkv=linear(_Q_RANK + 512, _HIDDEN),
        wq_b=linear(_HEADS * 512, _Q_RANK),
        wo_a=linear(_O_RANK, 8 * 512),
        wo_b=linear(_HIDDEN, _O_RANK),
        q_norm=norm(_Q_RANK),
        kv_norm=norm(512),
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


def moe(**changes):
    """A vLLM TP rank's MoE: FP4 routed experts as aiter holds them, the shared one an MXFP8 MLP."""

    def linear(rows, columns):
        return SimpleNamespace(
            weight=torch.zeros(rows, columns, dtype=torch.float8_e4m3fn),
            weight_scale=torch.zeros(rows // 32, columns // 32, dtype=torch.uint8),
        )

    def fp4(*shape):
        weight = torch.zeros(*shape, dtype=torch.uint8)
        weight.is_shuffled = True  # as vLLM's MXFP4 oracle marks the experts it shuffles for aiter
        return weight

    ffn = SimpleNamespace(
        gate=SimpleNamespace(
            weight=torch.zeros(_EXPERTS, _HIDDEN, dtype=torch.bfloat16),
            e_score_correction_bias=torch.zeros(_EXPERTS),
        ),
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(
                w13_weight=fp4(_EXPERTS, 2 * _INTER, _HIDDEN // 2),
                w13_weight_scale=torch.zeros(_EXPERTS * 2 * _INTER, _HIDDEN // 32, dtype=torch.uint8),
                w2_weight=fp4(_EXPERTS, _HIDDEN, _INTER // 2),
                w2_weight_scale=torch.zeros(_EXPERTS * _HIDDEN, 8, dtype=torch.uint8),
            )
        ),
        shared_experts=SimpleNamespace(gate_up_proj=linear(2 * 64, _HIDDEN), down_proj=linear(_HIDDEN, 64)),
        n_activated_experts=6,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )
    ffn.__dict__.update(changes)
    return ffn


def mono_decode(attn, ffn=None, ffn_only=False, first=False, tokens=64, stage_pre_mix=None):
    def norm():
        return SimpleNamespace(weight=torch.zeros(_HIDDEN, dtype=torch.bfloat16), variance_epsilon=1e-6)

    mono = SimpleNamespace(ffn_only=ffn_only)
    layer = SimpleNamespace(
        attn=attn,
        ffn=ffn or moe(),
        hc_mult=4,
        hc_attn_fn=torch.zeros(24, 4 * _HIDDEN),
        hc_attn_scale=torch.zeros(3),
        hc_attn_base=torch.zeros(24),
        attn_norm=norm(),
        hc_ffn_fn=torch.zeros(24, 4 * _HIDDEN),
        hc_ffn_scale=torch.zeros(3),
        hc_ffn_base=torch.zeros(24),
        ffn_norm=norm(),
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        engram=None,
    )
    x = torch.zeros(tokens, _HIDDEN, dtype=torch.bfloat16)
    positions = torch.zeros(tokens, dtype=torch.int64)
    if stage_pre_mix is not None:
        streams = torch.zeros(tokens, 4, _HIDDEN, dtype=torch.bfloat16)
        return api._mono_decode(mono, layer, streams, positions, None, None, None, stage_pre_mix)
    if first:
        return api._mono_decode(mono, layer, x, positions, None, None, None, None)
    return api._mono_decode(
        mono,
        layer,
        x,
        positions,
        torch.zeros(tokens, 4, _HIDDEN, dtype=torch.bfloat16),
        torch.zeros(tokens, 4, 1),
        torch.zeros(tokens, 4, 4),
        torch.zeros(tokens, 4),
    )


def test_a_decode_step_asks_for_the_layers_attention_front(asked, step):
    """K1 is asked for at the rank's heads with the post the previous layer owes, and the layer
    stays on vLLM."""
    assert mono_decode(attention()) is FallbackReason.NO_WORKLOAD
    (call,) = asked["deepseekv41_mono_k1"]
    assert call["residual"].shape == (64, 4, _HIDDEN)
    assert call["x"].shape == (64, _HIDDEN)
    assert call["post_mix"].shape == call["prev_pre_mix"].shape == (64, 4)
    assert call["comb_mix"].shape == (64, 4, 4)
    assert call["wq_b"].shape == (_HEADS * 512, _Q_RANK)
    assert call["positions"].shape == (64,)


def test_a_decode_step_vllms_kernels_take_asks_for_its_launches_at_one_ranks_shapes(asked, step):
    """48 rows are a step vLLM's mono kernels run: the front, the back with its FFN seam and the
    MoE are each asked for with what this rank holds, the other ranks' sum standing in by shape."""
    step.num_decodes = step.num_decode_tokens = 48
    assert mono_decode(attention(), tokens=48) is FallbackReason.NO_WORKLOAD
    assert [len(asked[name]) for name in _OPS[1:]] == [1, 1, 1]
    (back,), (ffn,) = asked["deepseekv41_mono_k2_attention"], asked["deepseekv41_mono_k2_ffn"]
    assert shapes(back) == {
        "q": (48, 1, _HEADS, 512),
        "kv_cache": (3, 32, 1, 584),
        "indices": (48, 1, 128),
        "sinks": (_HEADS,),
        "extra_kv_cache": (5, 64, 1, 584),
        "extra_indices": (48, 1, _TOPK),
        "rope_freqs": (16, 64),
        "positions": (48,),
        "wo_a": (_O_RANK, 8 * 512),
        "wo_a_scale": (_O_RANK // 32, 8 * 16),
        "wo_b": (_HIDDEN, _O_RANK),
        "wo_b_scale": (_HIDDEN // 32, _O_RANK // 32),
        "peer_x": (48, _HIDDEN),
        "residual": (48, 4, _HIDDEN),
        "post_mix": (48, 4),
        "comb_mix": (48, 4, 4),
        "pre_mix": (48, 4),
        "hc_fn": (24, 4 * _HIDDEN),
        "hc_scale": (3,),
        "hc_base": (24,),
        "norm_weight": (_HIDDEN,),
    }
    assert back["kv_cache_layout"] == "strided_pages"
    assert back["kv_cache"].stride() == (3 * 32 * 584, 584, 584, 1)
    assert shapes(ffn) == {
        "hidden_states": (48, _HIDDEN),
        "router_weight": (_EXPERTS, _HIDDEN),
        "correction_bias": (_EXPERTS,),
        "w1": (_EXPERTS, 2 * _INTER, _HIDDEN // 2),
        "w1_scale": (_EXPERTS * 2 * _INTER, _HIDDEN // 32),
        "w2": (_EXPERTS, _HIDDEN, _INTER // 2),
        "w2_scale": (_EXPERTS * _HIDDEN, 8),
        "shared_w1": (2 * 64, _HIDDEN),
        "shared_w1_scale": (4, _HIDDEN // 32),
        "shared_w2": (_HIDDEN, 64),
        "shared_w2_scale": (_HIDDEN // 32, 2),
    }
    # each call is one the op's contract keys, so one a capture records: a TP rank's variant
    assert point("deepseekv41_mono_k1", asked["deepseekv41_mono_k1"][0]) is not None
    back_point, ffn_point = (
        point("deepseekv41_mono_k2_attention", back),
        point("deepseekv41_mono_k2_ffn", ffn),
    )
    assert back_point.layout_variant_name == "strided_pages"
    assert back_point.attrs["has_wo_a_scale"] and back_point.attrs["has_peer_x"]
    assert {name: back_point.axes[name] for name in ("n_tokens", "n_heads", "o_dim")} == {
        "n_tokens": 48,
        "n_heads": _HEADS,
        "o_dim": _O_RANK,
    }
    assert ffn_point.layout_variant_name == "gate_up_interleaved" and ffn_point.attrs["has_shared_w1"]
    assert ffn_point.axes == {
        "n_tokens": 48,
        "model_dim": _HIDDEN,
        "n_routed": _EXPERTS,
        "inter_dim": _INTER,
        "shared_inter_dim": 64,
    }


def test_a_first_layers_attention_front_owes_no_post(asked, step):
    assert mono_decode(attention(), first=True) is FallbackReason.NO_WORKLOAD
    (call,) = asked["deepseekv41_mono_k1"]
    assert call["residual"].shape == (64, 4, _HIDDEN)
    # the contract's row_major storage: an expanded embedding (stride 0) is outside it
    assert call["residual"].is_contiguous()
    assert not {"x", "post_mix", "comb_mix", "prev_pre_mix"} & {k for k, v in call.items() if v is not None}


def test_a_pipeline_stages_first_layer_reads_its_streams_under_the_pre_mix_carried_in(asked, step):
    assert mono_decode(attention(), stage_pre_mix=torch.ones(64, 4)) is FallbackReason.NO_WORKLOAD
    (call,) = asked["deepseekv41_mono_k1"]
    assert call["residual"].shape == (64, 4, _HIDDEN)
    assert call["prev_pre_mix"].shape == (64, 4)
    assert not {"x", "post_mix", "comb_mix"} & {k for k, v in call.items() if v is not None}
    (ffn,) = asked["deepseekv41_mono_k2_ffn"]
    assert ffn["hidden_states"].shape == (64, _HIDDEN) and ffn["hidden_states"].is_contiguous()


@pytest.mark.parametrize(
    "layer",
    [{"ffn_only": True}, {"attn": attention(q_norm=SimpleNamespace(weight=None, variance_epsilon=1e-5))}],
    ids=["vllms_own_attention", "norms_of_two_epsilons"],
)
def test_a_layer_outside_the_attention_launches_is_asked_for_its_moe_alone(asked, step, layer):
    assert mono_decode(**{"attn": attention(), **layer}) is FallbackReason.NO_WORKLOAD
    assert list(asked) == ["deepseekv41_mono_k2_ffn"]


@pytest.mark.parametrize(
    ("layer", "prefills"),
    [
        ({}, 1),
        ({"ffn": moe(shared_experts=None)}, 0),
        ({"attn": attention(wo_a=None)}, 0),
        ({"attn": attention(n_local_heads=16)}, 0),
    ],
    ids=["a_step_with_prefills", "no_shared_mlp", "a_layer_read_otherwise", "heads_past_the_streams"],
)
def test_a_step_or_layer_the_route_does_not_know_stays_on_vllm_unasked_past_the_surprise(
    asked, step, layer, prefills
):
    """Nothing raises in the server: the route declines, with what it asked before the surprise."""
    step.num_prefills = prefills
    assert isinstance(mono_decode(**{"attn": attention(), **layer}), FallbackReason)
    assert "deepseekv41_mono_k2_attention" not in asked


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
