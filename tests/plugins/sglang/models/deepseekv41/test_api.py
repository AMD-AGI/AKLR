# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's DeepSeek-V4.1 routes hand the ops, on tensors shaped as sglang's."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseekv41 import api, contract


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


HIDDEN = 5120
# the rows a monokernel launch takes at most
MAX_ROWS = 48


def empty(*shape, dtype=torch.uint8):
    return torch.empty(shape, dtype=dtype)


@pytest.fixture
def asked(monkeypatch):
    """The Point each launch op is asked at, by its name (None: a call outside its contract):
    what a recorded call exports."""
    points = {}

    def record(op):
        codec, name = FastKeyCodec.from_contract(op), op.api_name.rpartition(".")[2]
        fast_key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()
        defaults = {param.name: param.default for param in op.api_params}

        def fallback_reason(**call):
            key = fast_key({**defaults, **call})
            points[name] = None if key is None else codec.to_point(key)
            return FallbackReason.NO_WORKLOAD

        monkeypatch.setattr(api, name, SimpleNamespace(fallback_reason=fallback_reason))

    for op in (
        contract.DEEPSEEKV41_MONO_K1,
        contract.DEEPSEEKV41_MONO_K2_ATTENTION,
        contract.DEEPSEEKV41_MONO_K2_FFN,
    ):
        record(op)
    return points


def monokernel_layer(tp, *, ratio=4, n_routed=8, topk=6):
    """A stand-in of the monokernel's layer (sgl_amd_mono's ``_Layer``) on one rank of a TP group of
    ``tp``, at DeepSeek-V4.1's widths but for ``n_routed`` experts, its caches not bound yet."""
    heads, groups, inter = 64 // tp, 8 // tp, {2: 1152, 4: 640}[tp]
    fp8, bf16, f32 = torch.float8_e4m3fn, torch.bfloat16, torch.float32
    k1 = SimpleNamespace(
        hc_fn=empty(24, 4 * HIDDEN, dtype=f32),
        hc_scale=empty(3, dtype=f32),
        hc_base=empty(24, dtype=f32),
        norm=empty(HIDDEN, dtype=bf16),
        wqkv=empty(1792, HIDDEN).view(fp8),
        wqkv_s=empty(56, HIDDEN // 32),
        q_norm=empty(1280, dtype=bf16),
        kv_norm=empty(512, dtype=bf16),
        wqb=empty(heads * 512, 1280).view(fp8),
        wqb_s=empty(heads * 16, 40),
    )
    k2 = SimpleNamespace(
        wob=empty(HIDDEN, groups * 1024).view(fp8),
        wob_s=empty(HIDDEN // 32, groups * 32),
        sink=empty(heads, dtype=f32),
    )
    ffn = SimpleNamespace(
        hc_ffn_fn=empty(24, 4 * HIDDEN, dtype=f32),
        hc_ffn_scale=empty(3, dtype=f32),
        hc_ffn_base=empty(24, dtype=f32),
        ffn_norm=empty(HIDDEN, dtype=bf16),
    )
    # the fused shared expert last, as sglang holds the experts for aiter: shuffled, the scales flat
    w13, w2 = empty(n_routed + 1, 2 * inter, HIDDEN // 2), empty(n_routed + 1, HIDDEN, inter // 2)
    w13.is_shuffled = w2.is_shuffled = True
    experts = SimpleNamespace(
        w13_weight=w13,
        w13_weight_scale_inv=empty((n_routed + 1) * 2 * inter, HIDDEN // 32),
        w2_weight=w2,
        w2_weight_scale_inv=empty((n_routed + 1) * HIDDEN, -(-inter // 256) * 8),
        moe_runner_config=SimpleNamespace(swiglu_limit=10.0),
    )
    routing = SimpleNamespace(correction_bias=empty(n_routed, dtype=bf16), top_k=topk + 1)
    moe = SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(weight=empty(n_routed, HIDDEN, dtype=bf16)),
        topk=SimpleNamespace(topk_config=routing),
        routed_scaling_factor=1.5,
    )
    attn = SimpleNamespace(
        eps=1e-6,
        freqs_cis=torch.zeros(4096, 32, dtype=torch.complex64),
        wo_a=SimpleNamespace(weight=empty(groups * 1024, 4096, dtype=bf16)),
    )
    module = SimpleNamespace(self_attn=attn, mlp=moe, rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20)
    return SimpleNamespace(
        i=3, module=module, k1=k1, k2=k2, ffn=ffn, ratio=ratio, swa=None, swa_block=0, comp=None, comp_block=0
    )


def monokernel_runner(tp):
    """A stand-in of the monokernel's runner on a rank of ``tp``: the buffers K1 writes for K2."""
    bf16, f32 = torch.bfloat16, torch.float32
    return SimpleNamespace(
        tp=tp,
        q=empty(MAX_ROWS, 64 // tp, 512, dtype=bf16),
        res_mid=empty(MAX_ROWS, 4, HIDDEN, dtype=bf16),
        post_a=empty(MAX_ROWS, 4, dtype=f32),
        comb_a=empty(MAX_ROWS, 4, 4, dtype=f32),
        pre_a=empty(MAX_ROWS, 4, dtype=f32),
        normed=empty(MAX_ROWS, HIDDEN, dtype=bf16),
    )


def kv_pool(page_size=64):
    """A pool of two pages as sglang holds it: a page's 584-byte slots in a row padded to 576s."""
    return empty(2, -(-page_size * 584 // 576) * 576)


@pytest.fixture
def forward_context(monkeypatch):
    """sglang's forward context of a step: the window's and the top-k's indices, and the pool, whose
    reads are counted."""
    reads = []
    metadata = SimpleNamespace(
        swa_page_indices=empty(MAX_ROWS, 128, dtype=torch.int32),
        sparse_page_indices=lambda ratio: empty(MAX_ROWS, 512, dtype=torch.int32),
    )
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(core_attn_metadata=metadata))
    pool = SimpleNamespace(
        get_swa_raw_buffer=lambda layer: reads.append(layer) or kv_pool(),
        get_swa_key_page_size=lambda: 64,
        get_extra_key_buffer=lambda layer: reads.append(layer) or kv_pool().view(torch.float8_e4m3fn),
        get_extra_key_page_size=lambda layer: 64,
    )
    module = ModuleType("sglang.srt.model_executor.forward_context")
    module.get_attn_backend = lambda: backend
    module.get_token_to_kv_pool = lambda: pool
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return reads


def owed_post(rows):
    """What a layer hands the next one's launch: (its MoE output, the streams, its post and
    combination mixes), then its pre mix."""
    f32 = torch.float32
    pending = (
        empty(rows, HIDDEN, dtype=torch.bfloat16),
        empty(rows, 4, HIDDEN, dtype=torch.bfloat16),
        empty(rows, 4, 1, dtype=f32),
        empty(rows, 4, 4, dtype=f32),
    )
    return pending, empty(rows, 4, dtype=f32)


@pytest.mark.parametrize(
    ("tp", "rows", "tok_par"),
    [(2, 5, False), (4, 48, True)],
    ids=["tp2_5_rows", "tp4_48_rows_token_parallel"],
)
def test_a_monokernel_launch_reaches_the_launch_ops_at_one_ranks_shapes(
    asked, forward_context, tp, rows, tok_par
):
    pending, pre = owed_post(rows)
    reason = api._launch(
        monokernel_runner(tp),
        monokernel_layer(tp),
        None,
        torch.arange(rows),
        None,
        pending,
        pre,
        tok_par=tok_par,
    )
    # the launch stays the monokernel's
    assert reason is FallbackReason.NO_WORKLOAD
    front, back, ffn = (
        asked[name]
        for name in ("deepseekv41_mono_k1", "deepseekv41_mono_k2_attention", "deepseekv41_mono_k2_ffn")
    )
    q_dim = 64 // tp * 512
    assert (
        front.axes.items()
        >= {"n_tokens": rows, "hidden_size": HIDDEN, "q_rank": 1280, "q_dim": q_dim}.items()
    )
    assert front.attrs == {
        "has_x": True,
        "has_post_mix": True,
        "has_comb_mix": True,
        "has_prev_pre_mix": True,
    }
    assert (
        back.axes.items()
        >= {
            "n_tokens": rows,
            "n_heads": 64 // tp,
            "page_size": 64,
            "n_indices": 128,
            "extra_page_size": 64,
            "n_extra_indices": 512,
            "hidden_size": HIDDEN,
            "o_dim": 8192 // tp,
        }.items()
    )
    assert back.attrs["has_peer_x"] and back.attrs["has_extra_kv_cache"]
    assert back.layout_variant_name == "padded_pages"
    # sglang pads a TP4 rank's 576 intermediate columns to 640
    assert ffn.axes == {
        "n_tokens": rows,
        "model_dim": HIDDEN,
        "n_routed": 8,
        "inter_dim": {2: 1152, 4: 640}[tp],
    }
    assert ffn.attrs == {"topk": 6} and ffn.layout_variant_name == "gate_up_interleaved"


def test_a_first_layers_launch_on_its_bound_window_cache_reads_no_pool(asked, forward_context):
    """No post owed, no previous pre mix, the embedding broadcast over the streams (a view), and
    a window-only layer whose cache its first launch bound."""
    layer = monokernel_layer(2, ratio=0)
    layer.swa, layer.swa_block = kv_pool(128), 128
    embedding = empty(3, 1, HIDDEN, dtype=torch.bfloat16).expand(3, 4, HIDDEN)
    api._launch(monokernel_runner(2), layer, None, torch.arange(3), None, (None, embedding, None, None), None)
    front, back = asked["deepseekv41_mono_k1"], asked["deepseekv41_mono_k2_attention"]
    assert front.axes["n_tokens"] == 3
    assert front.attrs == {
        "has_x": False,
        "has_post_mix": False,
        "has_comb_mix": False,
        "has_prev_pre_mix": False,
    }
    assert back.axes["page_size"] == 128 and not back.attrs["has_extra_kv_cache"]
    assert forward_context == []


def test_a_draft_stage_reaches_the_ffn_op_at_the_drafts_routing(asked, forward_context):
    pending, pre = owed_post(4)
    layer = monokernel_layer(2, n_routed=4, topk=3)
    api._launch(monokernel_runner(2), layer, None, torch.arange(4), None, pending, pre)
    assert asked["deepseekv41_mono_k2_ffn"].attrs == {"topk": 3}


def test_a_steps_positions_reach_the_front_launch_on_the_16_bytes_dispatch_requires(asked, forward_context):
    pending, pre = owed_post(4)
    # a step's positions as a slice of the batch's: int64, so 8 bytes past a boundary at an odd row
    api._launch(monokernel_runner(2), monokernel_layer(2), None, torch.arange(5)[1:], None, pending, pre)
    assert asked["deepseekv41_mono_k1"] is not None and asked["deepseekv41_mono_k2_attention"] is not None


def test_an_fp8_wo_a_leaves_the_back_launch_unasked(asked, forward_context):
    pending, pre = owed_post(4)
    layer = monokernel_layer(2)
    layer.module.self_attn.wo_a.weight = empty(4096, 4096).view(torch.float8_e4m3fn)
    api._launch(monokernel_runner(2), layer, None, torch.arange(4), None, pending, pre)
    assert sorted(asked) == ["deepseekv41_mono_k1", "deepseekv41_mono_k2_ffn"]


@pytest.mark.parametrize(
    "unexpected",
    [
        lambda layer, runner: delattr(layer, "k1"),
        lambda layer, runner: delattr(runner, "res_mid"),
        lambda layer, runner: setattr(layer.module.self_attn, "freqs_cis", torch.zeros(4096, 64)),
    ],
    ids=["no_front_weights", "no_seam_buffer", "a_real_rope_table"],
)
def test_a_launch_the_route_cannot_read_stays_the_monokernels(asked, forward_context, unexpected):
    """Another build's layer or runner: the route declines without raising in the server."""
    pending, pre = owed_post(4)
    layer, runner = monokernel_layer(2), monokernel_runner(2)
    unexpected(layer, runner)
    assert (
        api._launch(runner, layer, None, torch.arange(4), None, pending, pre)
        is FallbackReason.OUT_OF_CONTRACT
    )
