# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI
# Applies to the GDN route inventory addition.

"""What each route tells its op, and when it falls back to sglang."""

import sys
from types import SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.layers.dsa_backend.api as dsa_backend
import aklr.plugins.sglang.layers.norms.api as norms
import aklr.plugins.sglang.models.deepseek_v2.api as deepseek_v2
import aklr.plugins.sglang.models.minimax_m2.api as minimax_m2
import aklr.plugins.sglang.models.qwen2_moe.api as qwen2_moe
import aklr.plugins.sglang.models.qwen3_5.api as qwen3_5
import aklr.plugins.sglang.speculative.eagle_worker_v2.api as eagle
from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import hook
from aklr.plugins.sglang.models.deepseek_v2.contract import (
    DEEPSEEK_V2_ATTENTION,
    DEEPSEEK_V2_ATTENTION_INDEXER,
    DEEPSEEK_V2_DENSE_LAYER,
    DEEPSEEK_V2_EH_PROJ,
    DEEPSEEK_V2_FP8_ATTENTION_INDEXER,
    DEEPSEEK_V2_FP8_DENSE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART,
    DEEPSEEK_V2_MOE,
    DEEPSEEK_V2_MOE_LAYER,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART,
)
from aklr.plugins.sglang.speculative.eagle_worker_v2.contract import (
    EAGLE_WORKER_V2_DRAFT_TOP1,
)

FP8 = torch.float8_e4m3fn
ATTENTION_INDEXER_OPS = (
    "deepseek_v2_attention_indexer",
    "deepseek_v2_fp8_attention_indexer",
)


def meta(*shape, dtype=torch.bfloat16):
    """Shape and dtype without memory."""
    return torch.empty(shape, dtype=dtype, device="meta")


def contract_point(arguments, contract):
    """The Point of the call the route made, as dispatch keys it; None outside the contract."""
    codec = FastKeyCodec.from_contract(contract)
    call = {p.name: p.default for p in contract.api_params} | arguments
    key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()(call)
    return None if key is None else codec.to_point(key)


class Calls(list):
    """The calls made, and in ``asked`` the ops asked whether they serve."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = []


@pytest.fixture
def fake_ops(monkeypatch):
    """Every call a route makes to an op, as (op, arguments): a norm op and the draft top-1
    return "out"; the attention op stores 7s as the tokens' KV rows and returns the stream
    plus 2; an attention indexer op returns the stream plus 3 and its top-k of 5s; a MoE or
    dense layer op, or the MoE op, returns its input plus 1; eh_proj returns 5s. Asked first
    (fallback_reason), every op serves; what each was asked, as (op, arguments), in
    ``.asked``."""
    calls = Calls()

    def fake(module, name, result):
        def op(**arguments):
            calls.append((name, arguments))
            return result(**arguments)

        op.fallback_reason = lambda **arguments: calls.asked.append((name, arguments))
        monkeypatch.setattr(module, name, op)

    def attention(kv_cache, out_cache_loc, hidden_states, **_):
        kv_cache.view(torch.uint8)[out_cache_loc] = 7
        return hidden_states + 2

    for name in ("rms_norm", "rms_norm_add"):
        fake(norms, name, lambda **_: "out")
    fake(eagle, "eagle_worker_v2_draft_top1", lambda **_: "out")
    fake(deepseek_v2, "deepseek_v2_attention", attention)
    for name in ATTENTION_INDEXER_OPS:
        fake(
            deepseek_v2,
            name,
            lambda hidden_states, topk_indices, **_: (
                hidden_states + 3,
                topk_indices.fill_(5),
            ),
        )
    for name in (
        "deepseek_v2_moe",
        "deepseek_v2_moe_layer",
        "deepseek_v2_fp8_moe_layer",
        "deepseek_v2_fp8_moe_layer_shared_apart",
        "deepseek_v2_unquantized_moe_layer",
        "deepseek_v2_unquantized_moe_layer_shared_apart",
        "deepseek_v2_dense_layer",
        "deepseek_v2_fp8_dense_layer",
    ):
        fake(deepseek_v2, name, lambda hidden_states, **_: hidden_states + 1)
    fake(
        deepseek_v2,
        "deepseek_v2_eh_proj",
        lambda **_: torch.full((4, 256), 5.0, dtype=torch.bfloat16),
    )
    return calls


def norm_layer():
    return SimpleNamespace(weight=meta(6144), variance_epsilon=1e-5, _fused_pad_kernel=None)


def test_rms_norm_reaches_its_op(sglang, fake_ops):
    layer, x = norm_layer(), meta(4, 6144)
    assert norms._rms_norm_forward_aiter(layer, x) == "out"
    assert fake_ops == [("rms_norm", {"x": x, "weight": layer.weight, "eps": 1e-5})]


def test_rms_norm_add_reaches_its_op(sglang, fake_ops):
    layer, x, residual = norm_layer(), meta(4, 6144), meta(4, 6144)
    assert norms._rms_norm_forward_aiter(layer, x, residual, quant_linear=object()) == "out"
    assert fake_ops == [
        (
            "rms_norm_add",
            {"x": x, "residual": residual, "weight": layer.weight, "eps": 1e-5},
        )
    ]


@pytest.mark.parametrize("case", ["post_residual_addition", "fused-pad", "batch-invariant"])
def test_rms_norm_falls_back_to_sglang(sglang, fake_ops, case):
    layer, kwargs = norm_layer(), {}
    if case == "fused-pad":
        layer._fused_pad_kernel = object()
    elif case == "batch-invariant":
        sglang.batch_invariant = True
    else:
        kwargs[case] = object()
    result = norms._rms_norm_forward_aiter(layer, meta(4, 6144), **kwargs)
    assert isinstance(result, FallbackReason) and fake_ops == []


def runner():
    """aiter's fused_moe runner as sglang configures it for a DeepseekV2MoE."""
    return SimpleNamespace(
        config=SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        )
    )


def moe_layer():
    """sglang's DeepseekV2MoE on the TP group alone, shrunk: 8 routed experts of hidden and
    intermediate size 256, the shared one fused last, MXFP4 and shuffled as sglang stores
    them, on aiter's fused_moe runner; the bias fp32 as sglang keeps it, and none of its own
    for image tokens."""

    def stored(*shape):
        return torch.zeros(shape, dtype=torch.uint8)

    w13, w2 = stored(9, 512, 128), stored(9, 256, 128)
    w13.is_shuffled = w2.is_shuffled = True
    scheme = type("QuarkW4A4MXFp4MoE", (SimpleNamespace,), {})(runner=runner())
    experts = SimpleNamespace(
        scheme=scheme,
        w13_weight=w13,
        w13_weight_scale=stored(9, 512, 8),
        w2_weight=w2,
        w2_weight_scale=stored(9, 256, 8),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
    )
    routing = SimpleNamespace(
        top_k=9,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        num_fused_shared_experts=1,
        correction_bias=torch.zeros(8),
        scoring_func="sigmoid",
    )
    return SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(
            weight=torch.zeros(8, 256, dtype=torch.bfloat16),
            e_score_correction_bias_vl=None,
        ),
        topk=SimpleNamespace(topk_config=routing),
        routed_scaling_factor=2.5,
        is_hash=False,
        num_fused_shared_experts=1,
        moe_ep_size=1,
        _shared_expert_tp1=False,
    )


# sglang's DeepseekV2AttentionMLA, shrunk: hidden size 256, q_a 256, the kv latent 512 and
# k_pe 64, 2 heads of q_nope 192 and output 256; a KV cache of 16 slots
SLOTS, TOPK = torch.tensor([3, 4, 5, 6]), torch.zeros(4, 8, dtype=torch.int32)


def linear(rows, cols):
    """An fp8 linear with its block scales, its weight shuffled as sglang stores it."""
    weight = torch.zeros(rows, cols, dtype=FP8)
    weight.is_shuffled = True
    scale = torch.zeros(-(-rows // 128), cols // 128)
    return SimpleNamespace(weight=weight, weight_scale_inv=scale)


def norm(n=256):
    return SimpleNamespace(weight=torch.ones(n, dtype=torch.bfloat16), variance_epsilon=1e-5)


def attention_layer(runs_indexer=False):
    """The layer's state as the route reads it, its fp8 weights shuffled as sglang stores
    them."""

    def table():
        return torch.zeros(64, 1, 1, 32, dtype=torch.bfloat16)

    return SimpleNamespace(
        layer_id=11,
        use_dsa=True,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        next_skip_topk=True,
        fused_qkv_a_proj_with_mqa=linear(832, 256),
        q_a_layernorm=norm(),
        kv_a_layernorm=norm(512),
        q_b_proj=linear(512, 256),
        # sglang holds W_UK and W_UV transposed views of their absorbed layouts
        w_kc=torch.zeros(2, 512, 192, dtype=FP8).transpose(1, 2),
        w_vc=torch.zeros(2, 256, 512, dtype=FP8).transpose(1, 2),
        w_scale=torch.ones(()),
        o_proj=linear(256, 512),
        attn_mqa=SimpleNamespace(scaling=0.0625, k_scale_float=1.0),
        rotary_emb=SimpleNamespace(cos_cache=table(), sin_cache=table(), is_neox_style=False),
        should_run_indexer=lambda prev_topk_indices: runs_indexer,
        dispatch_attn_forward_method=lambda forward_batch: "mla_rocm",
    )


def target_verify(sglang):
    """A target-verify batch of 4 tokens on the DSA backend's fp8 KV cache of 16 slots, not
    split into two-batch overlap's microbatches."""
    sglang.backend = SimpleNamespace(use_fused_topk=True, hisparse_coordinator=None)
    sglang.kv = torch.zeros(16, 1, 576, dtype=FP8)
    sglang.pool = SimpleNamespace(
        dsa_kv_cache_store_fp8=False,
        kv_cache_dim=576,
        get_key_buffer=lambda layer: sglang.kv,
    )
    mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: True,
        is_draft_extend_v2=lambda: False,
    )
    return SimpleNamespace(forward_mode=mode, out_cache_loc=SLOTS, can_run_tbo=False)


def draft_extend(sglang):
    """target_verify()'s batch as the MTP draft's extend of its 4 tokens."""
    forward_batch = target_verify(sglang)
    forward_batch.forward_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: True,
    )
    return forward_batch


def boundary(module):
    """The stand-in of one of sglang's layer_boundary modules."""
    return sys.modules[f"sglang.srt.layers.layer_boundary.{module}"]


def decoder_layer(runs_indexer=False):
    """sglang's DeepseekV2DecoderLayer of an MoE layer, shrunk: attention_layer()'s attention,
    moe_layer()'s MoE, both RMSNorms, and the attention's boundary, on the ordinary batch
    with its input on all of the batch's rows."""
    return SimpleNamespace(
        self_attn=attention_layer(runs_indexer),
        mlp=moe_layer(),
        is_layer_sparse=True,
        input_layernorm=norm(),
        post_attention_layernorm=norm(),
        attn_boundary=SimpleNamespace(
            plan=SimpleNamespace(variant_for=lambda batch: boundary("contracts").BatchVariant.ORDINARY),
            input_on_attn_tp_slices=False,
        ),
    )


def dense_layer():
    """decoder_layer() as a dense layer: sglang's DeepseekV2MLP on the TP group, bf16 weights."""
    layer = decoder_layer()
    layer.is_layer_sparse = False
    layer.mlp = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.zeros(256, 256, dtype=torch.bfloat16)),
        down_proj=SimpleNamespace(weight=torch.zeros(256, 128, dtype=torch.bfloat16)),
        swiglu_limit=None,
        is_replicated=False,
    )
    return layer


def run_layer(layer, forward_batch, hidden=None, **kwargs):
    """The layer route on hidden (default 2s) and the batch's residual stream (default the
    one a layer of the route's own leaves: hidden written as its residual), and its
    result."""
    if hidden is None:
        hidden = torch.full((4, 256), 2.0, dtype=torch.bfloat16)
    if not hasattr(forward_batch, "residual_stream"):
        stream = boundary("residual.stream").ResidualStream(hidden)
        forward_batch.residual_stream = stream
    return hidden, deepseek_v2._decoder_layer(layer, torch.arange(4), hidden, forward_batch, None, **kwargs)


PLAIN_ADD = SimpleNamespace(is_plain_add=True)


def after_sglang_layer(forward_batch, owes_sum=False):
    """The batch's residual stream as a layer of sglang's own leaves it: the residual 1s and
    the MoE's output 2s pending on it, its sum over the ranks still owed if ``owes_sum`` (an
    all-reduce over 8 ranks in place, each recorded in forward_batch.all_reduced); and the
    handle that layer returns."""
    stream = boundary("residual.stream").ResidualStream(torch.ones(4, 256, dtype=torch.bfloat16))
    output = torch.full((4, 256), 2.0, dtype=torch.bfloat16)
    forward_batch.all_reduced = []
    if owes_sum:
        group = SimpleNamespace(
            all_reduce=lambda partial: forward_batch.all_reduced.append(partial) or partial.mul_(8)
        )
        output = boundary("output").UnreducedOutput(output, group)
    forward_batch.residual_stream = stream
    return stream.record(output, PLAIN_ADD)


def test_layer_runs_its_ops_on_the_residual_stream(sglang, fake_ops):
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    handle = after_sglang_layer(forward_batch)
    residual_stream = forward_batch.residual_stream
    _, (out, topk) = run_layer(layer, forward_batch, handle, prev_topk_indices=TOPK)
    (attention, arguments), (op, moe_arguments) = fake_ops
    assert (attention, op) == ("deepseek_v2_attention", "deepseek_v2_moe_layer")
    for call, contract in (
        (arguments, DEEPSEEK_V2_ATTENTION),
        (moe_arguments, DEEPSEEK_V2_MOE_LAYER),
    ):
        point = contract_point(call, contract)
        assert point is not None and point.layout_variant_name == "shuffled_16x16"
    assert (moe_arguments["topk"], moe_arguments["routed_scaling_factor"]) == (8, 2.5)
    # the stream is the residual plus the output pending on it, normed in each op and added
    # to after it
    stream = arguments["hidden_states"]
    assert (stream == 1 + 2).all()
    assert arguments["input_norm_weight"] is layer.input_layernorm.weight
    assert moe_arguments["norm_weight"] is layer.post_attention_layernorm.weight
    assert torch.equal(moe_arguments["hidden_states"], stream + 2)
    # the op's kv_cache is sglang's cache: the rows land at their slots, and only there
    stored = sglang.kv.view(16, 576).view(torch.uint8)
    assert (stored[SLOTS] == 7).all() and stored.sum() == 7 * 4 * 576
    # it leaves written back as the residual with nothing pending, the top-k carried on
    assert arguments["topk_indices"] is TOPK
    assert torch.equal(out, stream + 3)
    assert residual_stream.residual is out and residual_stream.pending is None
    assert topk is TOPK


@pytest.mark.parametrize("written", [True, False], ids=["written", "stack-input"])
def test_layer_takes_a_whole_stream_as_it_is(sglang, fake_ops, written):
    """The residual a layer of the route's own wrote, or the layer stack's input: the layer
    writes the stream after it back as the residual, with nothing pending."""
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    hidden = torch.full((4, 256), 2.0, dtype=torch.bfloat16)
    stream = boundary("residual.stream").ResidualStream(hidden if written else None)
    forward_batch.residual_stream = stream
    _, (out, topk) = run_layer(layer, forward_batch, hidden, prev_topk_indices=TOPK)
    (_, arguments), (_, moe_arguments) = fake_ops
    assert arguments["hidden_states"] is hidden
    assert torch.equal(out, moe_arguments["hidden_states"] + 1)
    assert stream.residual is out and stream.pending is None and topk is TOPK


@pytest.mark.parametrize("owes_sum", [False, True], ids=["summed", "owes-sum"])
def test_layer_takes_the_output_pending_on_the_stream(sglang, fake_ops, owes_sum):
    """After a layer of sglang's own: the residual plus that layer's output, its sum completed
    on a copy; the stream after the layer written back, the output released."""
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    handle = after_sglang_layer(forward_batch, owes_sum)
    stream = forward_batch.residual_stream
    contribution, partial = stream.pending, stream.pending.value
    _, (out, topk) = run_layer(layer, forward_batch, handle, prev_topk_indices=TOPK)
    (_, arguments), (_, moe_arguments) = fake_ops
    assert (arguments["hidden_states"] == (2 * 8 if owes_sum else 2) + 1).all()
    assert (partial == 2).all()
    assert torch.equal(out, moe_arguments["hidden_states"] + 1)
    assert stream.residual is out and stream.pending is None and topk is TOPK
    assert contribution.value is None and contribution.owed is None


@pytest.fixture
def captured(monkeypatch):
    """The layers whose top-k sglang's indexer capturer records."""
    layers = []
    monkeypatch.setattr(
        sys.modules["sglang.srt.state_capturer.indexer_topk"],
        "maybe_capture_indexer_topk",
        lambda layer_id, topk: layers.append(layer_id) or topk,
    )
    return layers


def test_layer_captures_the_carried_topk_once_its_op_serves(sglang, fake_ops, captured):
    run_layer(decoder_layer(), target_verify(sglang), prev_topk_indices=TOPK)
    assert captured == [11]


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        ("deepseek_v2_attention", FallbackReason.NO_SCHEMA),
        ("deepseek_v2_moe_layer", FallbackReason.NO_WORKLOAD),
    ],
    ids=["attention", "moe"],
)
def test_layer_falls_back_whole_before_anything_runs(sglang, fake_ops, captured, monkeypatch, op, reason):
    """Either op asked first that it falls back: sglang runs the whole layer on the stream as
    its last layer left it, the sum that layer owes not completed here; no op runs and no
    top-k is captured."""

    def ran(**_):
        raise AssertionError(f"{op} ran")

    ran.fallback_reason = lambda **_: reason
    monkeypatch.setattr(deepseek_v2, op, ran)
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    handle = after_sglang_layer(forward_batch, owes_sum=True)
    stream = forward_batch.residual_stream
    residual, contribution = stream.residual, stream.pending
    owed = contribution.owed
    _, result = run_layer(layer, forward_batch, handle, prev_topk_indices=TOPK)
    assert result is reason
    assert fake_ops == [] and captured == [] and forward_batch.all_reduced == []
    assert stream.residual is residual and (residual == 1).all()
    assert stream.pending is contribution and contribution.owed is owed
    assert (contribution.value == 2).all()


@pytest.mark.parametrize("indexer", [False, True], ids=["carried", "indexer"])
def test_layer_asks_its_ops_on_a_tensor_shaped_as_its_input(sglang, aiter, fake_ops, indexer):
    """After a layer of sglang's own that owes its sum, hidden_states is that layer's opaque
    handle: both ops are asked on the output pending on the residual stream, a tensor shaped
    as the layer's input, and the sum completes once, after both serve."""
    if indexer:
        layer, forward_batch, _ = attention_indexer_layer(sglang)
        kwargs = {}
    else:
        layer, forward_batch = decoder_layer(), target_verify(sglang)
        kwargs = {"prev_topk_indices": TOPK}
    handle = after_sglang_layer(forward_batch, owes_sum=True)
    assert not torch.is_tensor(handle)
    run_layer(layer, forward_batch, handle, **kwargs)
    asked = [arguments["hidden_states"] for _, arguments in fake_ops.asked]
    assert len(asked) == 2
    assert all(torch.is_tensor(h) and h.shape == (4, 256) for h in asked)
    assert len(forward_batch.all_reduced) == 1 and len(fake_ops) == 2


def test_layer_falls_back_whole_when_its_moe_op_does_once_called(sglang, fake_ops, monkeypatch):
    """A MoE op that served when asked but falls back when called, after the attention op:
    sglang runs the whole layer again on the stream its last layer left, the sum that layer
    owes still to complete once."""
    moe = deepseek_v2.deepseek_v2_moe_layer

    def late(**arguments):
        moe(**arguments)
        return FallbackReason.NO_WORKLOAD

    late.fallback_reason = lambda **_: None
    monkeypatch.setattr(deepseek_v2, "deepseek_v2_moe_layer", late)
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    handle = after_sglang_layer(forward_batch, owes_sum=True)
    stream = forward_batch.residual_stream
    residual, contribution = stream.residual, stream.pending
    owed = contribution.owed
    _, result = run_layer(layer, forward_batch, handle, prev_topk_indices=TOPK)
    assert result is FallbackReason.NO_WORKLOAD
    assert [op for op, _ in fake_ops] == [
        "deepseek_v2_attention",
        "deepseek_v2_moe_layer",
    ]
    assert stream.residual is residual and (residual == 1).all()
    assert stream.pending is contribution and contribution.owed is owed
    assert (contribution.value == 2).all()
    stream.check(handle)


def prefill(s, layer, batch):
    batch.forward_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )


def lora(s, layer, batch):
    # sglang's FusedMoEWithLoRA copies quant_method, not scheme.
    del layer.mlp.experts.scheme
    layer.mlp.experts.quant_method = type("QuarkFusedMoEMethod", (), {})()


def mxfp4_runner(layer):
    return layer.mlp.experts.scheme.runner


# each: change(sglang, layer, forward batch) that makes the route fall back
LAYER_FALLBACKS = {
    "prefill": prefill,
    "other-forward-method": lambda s, layer, batch: setattr(
        layer.self_attn, "dispatch_attn_forward_method", lambda forward_batch: "mla"
    ),
    "topk-of-positions": lambda s, layer, batch: setattr(s.backend, "use_fused_topk", False),
    "fp8-store-rows": lambda s, layer, batch: setattr(s.pool, "dsa_kv_cache_store_fp8", True),
    "scaled-kv": lambda s, layer, batch: setattr(layer.self_attn.attn_mqa, "k_scale_float", 0.5),
    "neox-rope": lambda s, layer, batch: setattr(layer.self_attn.rotary_emb, "is_neox_style", True),
    "lora": lambda s, layer, batch: setattr(s, "lora", True),
    "dcp": lambda s, layer, batch: setattr(s.parallel, "dcp_enabled", True),
    "experts-lora": lora,
    "hash-routing": lambda s, layer, batch: setattr(layer.mlp, "is_hash", True),
    "vision-bias": lambda s, layer, batch: setattr(
        layer.mlp.gate, "e_score_correction_bias_vl", torch.zeros(8)
    ),
    "deterministic": lambda s, layer, batch: setattr(
        s.config.deterministic, "enable_deterministic_inference", True
    ),
    "eplb": lambda s, layer, batch: setattr(s.config.moe, "enable_eplb", True),
    "recorder": lambda s, layer, batch: setattr(s.config.moe, "expert_distribution_recorder_mode", "stat"),
    "return-routed-experts": lambda s, layer, batch: setattr(
        s.config.features, "enable_return_routed_experts", True
    ),
    "shared-expert-apart": lambda s, layer, batch: setattr(layer.mlp, "num_fused_shared_experts", 0),
    "no-aiter-runner": lambda s, layer, batch: setattr(layer.mlp.experts.scheme, "runner", None),
    "gelu": lambda s, layer, batch: setattr(mxfp4_runner(layer).config, "activation", "gelu"),
    "expert-mask": lambda s, layer, batch: setattr(
        layer.mlp.experts.dispatcher, "expert_mask_gpu", torch.ones(8)
    ),
    "expert-parallel": lambda s, layer, batch: setattr(layer.mlp, "moe_ep_size", 2),
    "dp-attention": lambda s, layer, batch: setattr(s, "dp_attention", True),
}


@pytest.mark.parametrize("change", LAYER_FALLBACKS.values(), ids=list(LAYER_FALLBACKS))
def test_layer_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    change(sglang, layer, forward_batch)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def test_layer_without_a_carried_topk_falls_back(sglang, fake_ops):
    """A shared layer reached with no top-k (sglang attends dense then)."""
    _, result = run_layer(decoder_layer(), target_verify(sglang))
    assert isinstance(result, FallbackReason) and fake_ops == []


def test_layer_falls_back_on_a_capture(sglang, fake_ops):
    _, result = run_layer(
        decoder_layer(),
        target_verify(sglang),
        prev_topk_indices=TOPK,
        captured_last_layer_outputs=object(),
    )
    assert isinstance(result, FallbackReason) and fake_ops == []


def batch_variant(name):
    return lambda forward_batch: getattr(boundary("contracts").BatchVariant, name)


# each: change(sglang, layer, forward batch) that makes the route fall back on the stream
# after_sglang_layer() leaves
STREAM_FALLBACKS = {
    "no-stream": lambda s, layer, batch: setattr(batch, "residual_stream", None),
    # a stage's last layer would hand the next stage no residual, which it reads
    "pipeline": lambda s, layer, batch: setattr(s.parallel, "pp_size", 2),
    "two-batch-overlap": lambda s, layer, batch: setattr(batch, "can_run_tbo", True),
    "input-scattered": lambda s, layer, batch: setattr(
        layer.attn_boundary.plan, "variant_for", batch_variant("INPUT_SCATTERED")
    ),
    "context-parallel": lambda s, layer, batch: setattr(
        layer.attn_boundary.plan, "variant_for", batch_variant("CONTEXT_PARALLEL")
    ),
    "attention-on-tp-slices": lambda s, layer, batch: setattr(
        layer.attn_boundary, "input_on_attn_tp_slices", True
    ),
    "attention-tp-short-of-tp": lambda s, layer, batch: setattr(s.parallel, "attn_tp_size", 4),
    "deferred-finalize": lambda s, layer, batch: setattr(
        batch.residual_stream.pending, "owed", boundary("output").DeferredFinalize()
    ),
    "not-plain-add": lambda s, layer, batch: setattr(
        batch.residual_stream.pending, "update", SimpleNamespace(is_plain_add=False)
    ),
}


@pytest.mark.parametrize("change", STREAM_FALLBACKS.values(), ids=list(STREAM_FALLBACKS))
def test_layer_falls_back_before_taking_the_stream(sglang, fake_ops, change):
    layer, forward_batch = decoder_layer(), target_verify(sglang)
    handle = after_sglang_layer(forward_batch)
    stream = forward_batch.residual_stream
    residual, contribution = stream.residual, stream.pending
    change(sglang, layer, forward_batch)
    _, result = run_layer(layer, forward_batch, handle, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []
    assert stream.residual is residual and stream.pending is contribution
    assert contribution.value is handle


def test_dense_layer_runs_its_mlp_op_on_the_stream(sglang, fake_ops):
    layer, forward_batch = dense_layer(), target_verify(sglang)
    _, (out, topk) = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    (attention, _), (op, arguments) = fake_ops
    assert (attention, op) == ("deepseek_v2_attention", "deepseek_v2_dense_layer")
    assert contract_point(arguments, DEEPSEEK_V2_DENSE_LAYER) is not None
    assert arguments["norm_weight"] is layer.post_attention_layernorm.weight
    assert arguments["gate_up_weight"] is layer.mlp.gate_up_proj.weight
    assert arguments["down_weight"] is layer.mlp.down_proj.weight
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK
    assert forward_batch.residual_stream.residual is out


DENSE_FALLBACKS = {
    "fp8-gate-up-alone": lambda s, layer, batch: setattr(
        layer.mlp.gate_up_proj, "weight", torch.zeros(256, 256, dtype=FP8)
    ),
    "swiglu-limit": lambda s, layer, batch: setattr(layer.mlp, "swiglu_limit", 7.0),
    "nvfp4-fusion": lambda s, layer, batch: setattr(layer.mlp, "_enable_nvfp4_gemm_swiglu_fusion", True),
    # each rank runs the MLP whole (moe_dense_tp_size 1)
    "replicated": lambda s, layer, batch: setattr(layer.mlp, "is_replicated", True),
}


@pytest.mark.parametrize("change", DENSE_FALLBACKS.values(), ids=list(DENSE_FALLBACKS))
def test_dense_layer_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = dense_layer(), target_verify(sglang)
    change(sglang, layer, forward_batch)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def fp8_dense_layer():
    """dense_layer() of GLM-5.3-FP8: its MLP's linears fp8 with their block scales, shuffled
    as sglang stores them."""
    layer = dense_layer()
    layer.mlp.gate_up_proj = linear(256, 256)
    layer.mlp.down_proj = linear(256, 128)
    return layer


def test_fp8_dense_layer_runs_its_mlp_op(sglang, fake_ops):
    layer = fp8_dense_layer()
    _, (out, topk) = run_layer(layer, target_verify(sglang), prev_topk_indices=TOPK)
    (attention, _), (op, arguments) = fake_ops
    assert (attention, op) == ("deepseek_v2_attention", "deepseek_v2_fp8_dense_layer")
    point = contract_point(arguments, DEEPSEEK_V2_FP8_DENSE_LAYER)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    gate_up, down = layer.mlp.gate_up_proj, layer.mlp.down_proj
    assert arguments["gate_up_weight"] is gate_up.weight
    assert arguments["gate_up_scale"] is gate_up.weight_scale_inv
    assert arguments["down_weight"] is down.weight
    assert arguments["down_scale"] is down.weight_scale_inv
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK


def test_fp8_dense_layer_with_a_bf16_down_falls_back(sglang, fake_ops):
    layer = fp8_dense_layer()
    layer.mlp.down_proj.weight = torch.zeros(256, 128, dtype=torch.bfloat16)
    _, result = run_layer(layer, target_verify(sglang), prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def dsa(module):
    """The stand-in of one of sglang's DSA modules."""
    return sys.modules[f"sglang.srt.layers.attention.dsa.{module}"]


@pytest.fixture
def aiter(sglang, monkeypatch):
    """sglang on HIP, its indexer on aiter's preshuffled paged kernels."""
    for name in ("_is_hip", "_use_aiter", "_use_aiter_preshuffle"):
        monkeypatch.setattr(dsa("dsa_indexer"), name, True)


def attention_indexer_layer(sglang, fp8=True):
    """decoder_layer(runs_indexer=True) where deepseek_v2_fp8_attention_indexer serves it (with
    bf16 wq_b and wk, deepseek_v2_attention_indexer), and its DSA metadata: the indexer's 4 heads
    of 128 and a top-k of 8 on the fused paged v2 kernel, each token's 60 positions on one page
    of 64 slots (KV rows and index K, the index K not from aiter's fused fp8 writer)."""

    def bf16(rows, cols):
        return SimpleNamespace(weight=torch.zeros(rows, cols, dtype=torch.bfloat16))

    layer, forward_batch = decoder_layer(runs_indexer=True), target_verify(sglang)
    forward_batch.seq_lens = torch.tensor([60])
    k_norm = sys.modules["sglang.srt.layers.layernorm"].LayerNorm()
    k_norm.weight = k_norm.bias = torch.zeros(128, dtype=torch.bfloat16)
    k_norm.variance_epsilon = 1e-6
    layer.self_attn.indexer = dsa("dsa_indexer").Indexer(
        use_dsa_indexer_fusion=False,
        dsa_enable_prefill_cp=False,
        wq_b=(linear if fp8 else bf16)(512, 256),
        wk=(linear if fp8 else bf16)(128, 256),
        weights_proj=bf16(4, 256),
        scale_fmt="ue8m0",
        block_size=128,
        head_dim=128,
        k_norm=k_norm,
        rotary_emb=SimpleNamespace(
            cos_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
            sin_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
            is_neox_style=False,
        ),
        _should_skip_logits_computation=lambda batch: False,
        paged_mqa_logits_backend=SimpleNamespace(is_aiter=lambda: True),
        num_init_tokens=0,
        num_local_tokens=0,
        index_topk=8,
        use_aiter_fused_fp8_writer=False,
    )
    metadata = dsa("dsa_indexer_metadata").DSAIndexerMetadata(
        attn_metadata=SimpleNamespace(page_size=64, real_page_table=torch.zeros(4, 1, dtype=torch.int32)),
        force_unfused_topk=False,
        topk_backend=SimpleNamespace(should_use_topk_v2=lambda: True),
        topk_transform_method=dsa("dsa_topk_backend").TopkTransformMethod.PAGED,
        get_seqlens_expanded=lambda: torch.full((4,), 60, dtype=torch.int32),
        get_page_table_64=lambda: torch.zeros(4, 1, dtype=torch.int32),
    )
    sglang.backend.get_indexer_metadata = lambda layer_id, batch: metadata
    sglang.kv = torch.zeros(64, 1, 576, dtype=FP8)
    sglang.index_k = torch.zeros(1, 64 * (128 + 4), dtype=torch.uint8)
    sglang.pool.page_size = sglang.pool.index_page_size = 64
    sglang.pool.get_index_k_with_scale_buffer = lambda layer_id: sglang.index_k
    return layer, forward_batch, metadata


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
def test_layer_attention_and_indexer_are_one_op_call(sglang, aiter, fake_ops, fp8):
    """One op call for the attention and the indexer: the fp8 op, its indexer scales passed,
    for fp8 wq_b and wk; else the op of bf16 ones."""
    layer, forward_batch, _ = attention_indexer_layer(sglang, fp8)
    handle = after_sglang_layer(forward_batch)
    _, (out, topk) = run_layer(layer, forward_batch, handle)
    (op, arguments), (moe_op, moe_arguments) = fake_ops
    assert (op, moe_op) == (ATTENTION_INDEXER_OPS[fp8], "deepseek_v2_moe_layer")
    contract = (DEEPSEEK_V2_ATTENTION_INDEXER, DEEPSEEK_V2_FP8_ATTENTION_INDEXER)[fp8]
    point = contract_point(arguments, contract)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    assert (arguments["hidden_states"] == 1 + 2).all()
    assert arguments["input_norm_weight"] is layer.input_layernorm.weight
    indexer = layer.self_attn.indexer
    if fp8:
        assert arguments["wq_b_scale"] is indexer.wq_b.weight_scale_inv
        assert arguments["wk_scale"] is indexer.wk.weight_scale_inv
    else:
        assert "wq_b_scale" not in arguments and "wk_scale" not in arguments
    # the op stores into sglang's caches
    assert arguments["kv_cache"].data_ptr() == sglang.kv.data_ptr()
    assert arguments["index_k_cache"].data_ptr() == sglang.index_k.data_ptr()
    # the MoE takes the stream after the attention, and the top-k is carried on
    assert torch.equal(moe_arguments["hidden_states"], arguments["hidden_states"] + 3)
    assert torch.equal(out, moe_arguments["hidden_states"] + 1)
    assert topk is arguments["topk_indices"] and (topk == 5).all()


def attention_indexer(layer, forward_batch):
    """_attention_indexer_call on a stream of 0s."""
    attn = layer.self_attn
    args = deepseek_v2._attention_args(
        attn,
        torch.zeros(4, 256, dtype=torch.bfloat16),
        layer.input_layernorm,
        torch.arange(4),
        forward_batch,
    )
    return deepseek_v2._attention_indexer_call(attn, args, forward_batch)


# each: change(sglang, layer, metadata) that leaves the indexer to sglang
ATTENTION_INDEXER_FALLBACKS = {
    "topk-broadcast": lambda s, layer, metadata: setattr(
        sys.modules["sglang.srt.environ"].envs,
        "SGLANG_DSA_TOPK_BROADCAST",
        SimpleNamespace(get=lambda: True),
    ),
    "unfused-topk": lambda s, layer, metadata: setattr(metadata, "force_unfused_topk", True),
    "ragged-topk": lambda s, layer, metadata: setattr(
        metadata,
        "topk_transform_method",
        dsa("dsa_topk_backend").TopkTransformMethod.RAGGED,
    ),
    "topk-past-2048": lambda s, layer, metadata: setattr(layer.self_attn.indexer, "index_topk", 4096),
    "local-tokens": lambda s, layer, metadata: setattr(layer.self_attn.indexer, "num_local_tokens", 128),
    "other-page-size": lambda s, layer, metadata: setattr(s.pool, "page_size", 1),
    "other-index-page-size": lambda s, layer, metadata: setattr(s.pool, "index_page_size", 32),
    # SWAKVPool's, which states no index page size
    "no-index-page-size": lambda s, layer, metadata: delattr(s.pool, "index_page_size"),
    # storing index K unrotated, from an fp32 k_norm
    "fused-fp8-writer": lambda s, layer, metadata: setattr(
        layer.self_attn.indexer, "use_aiter_fused_fp8_writer", True
    ),
    "layer-split-pool": lambda s, layer, metadata: setattr(
        s.pool, "get_broadcastable_index_k_with_scale_buffer", None
    ),
    "no-dsa-metadata": lambda s, layer, metadata: setattr(
        s.backend, "get_indexer_metadata", lambda layer_id, batch: None
    ),
    # IndexerKPool's, which has none of Indexer's paths: falls back without reading them
    "other-indexer": lambda s, layer, metadata: setattr(
        layer.self_attn, "indexer", SimpleNamespace(index_topk=8)
    ),
}


@pytest.mark.parametrize(
    "change",
    ATTENTION_INDEXER_FALLBACKS.values(),
    ids=list(ATTENTION_INDEXER_FALLBACKS),
)
def test_attention_indexer_falls_back_to_sglang(sglang, aiter, fake_ops, change):
    """The reason, and the layer whole to sglang."""
    layer, forward_batch, metadata = attention_indexer_layer(sglang)
    change(sglang, layer, metadata)
    assert isinstance(attention_indexer(layer, forward_batch), FallbackReason)
    _, result = run_layer(layer, forward_batch)
    assert isinstance(result, FallbackReason) and fake_ops == []


def unquantized_experts(n_experts=9):
    """sglang's FusedMoE on UnquantizedFusedMoEMethod's aiter runner: ``n_experts`` bf16
    experts of hidden size 256 and intermediate size 32 (the router's 8, then the shared one
    fused last), shuffled for the runner without shuffle_weight's mark."""
    backend = SimpleNamespace(is_triton_kernels=lambda: False, is_deep_gemm=lambda: False)
    method = type("UnquantizedFusedMoEMethod", (SimpleNamespace,), {})(
        _aiter_runner=runner(),
        runner=SimpleNamespace(runner_backend=backend),
        use_flashinfer_cutlass=False,
        use_flashinfer_trtllm_moe=False,
    )
    bf16 = torch.bfloat16
    return SimpleNamespace(
        quant_method=method,
        w13_weight=torch.zeros(n_experts, 64, 256, dtype=bf16),
        w2_weight=torch.zeros(n_experts, 256, 32, dtype=bf16),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
        _skip_aiter_moe_shuffle=False,
    )


def draft_decoder_layer():
    """decoder_layer() as GLM-5.2's MTP draft layer on sglang 0.5.22: W_UK and W_UV bf16
    (sglang quantizes the main model's alone, leaving this layer's scale the float 1.0), and
    unquantized_experts(): the router's 8, then the shared expert fused last."""
    layer, bf16 = decoder_layer(), torch.bfloat16
    attn = layer.self_attn
    attn.w_kc = torch.zeros(2, 512, 192, dtype=bf16).transpose(1, 2)
    attn.w_vc = torch.zeros(2, 256, 512, dtype=bf16).transpose(1, 2)
    attn.w_scale = 1.0
    layer.mlp.experts = unquantized_experts()
    return layer


def test_draft_layer_runs_its_ops_with_the_shared_expert_fused(sglang, fake_ops):
    layer = draft_decoder_layer()
    _, (out, topk) = run_layer(layer, draft_extend(sglang), prev_topk_indices=TOPK)
    (attention, attention_arguments), (op, arguments) = fake_ops
    assert (attention, op) == (
        "deepseek_v2_attention",
        "deepseek_v2_unquantized_moe_layer",
    )
    # its bf16 W_UK and W_UV come without a scale
    assert attention_arguments["w_scale"] is None
    assert contract_point(attention_arguments, DEEPSEEK_V2_ATTENTION) is not None
    point = contract_point(arguments, DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    experts = layer.mlp.experts
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w2"] is experts.w2_weight
    assert (arguments["topk"], arguments["routed_scaling_factor"]) == (8, 2.5)
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK


def draft_decoder_layer_shared_apart():
    """draft_decoder_layer() with the shared expert apart: unquantized_experts() of the
    router's 8 alone, the shared expert a bf16 DeepseekV2MLP of intermediate size 32."""
    layer, bf16 = draft_decoder_layer(), torch.bfloat16
    mlp, routing = layer.mlp, layer.mlp.topk.topk_config
    mlp.experts, mlp.num_fused_shared_experts = unquantized_experts(8), 0
    routing.top_k, routing.num_fused_shared_experts = 8, 0
    mlp.shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.zeros(64, 256, dtype=bf16)),
        down_proj=SimpleNamespace(weight=torch.zeros(256, 32, dtype=bf16)),
        swiglu_limit=None,
    )
    return layer


def test_draft_layer_runs_its_ops_with_the_shared_expert_apart(sglang, fake_ops):
    layer = draft_decoder_layer_shared_apart()
    _, (out, topk) = run_layer(layer, draft_extend(sglang), prev_topk_indices=TOPK)
    (attention, _), (op, arguments) = fake_ops
    assert (attention, op) == (
        "deepseek_v2_attention",
        "deepseek_v2_unquantized_moe_layer_shared_apart",
    )
    point = contract_point(arguments, DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    experts, shared = layer.mlp.experts, layer.mlp.shared_experts
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w2"] is experts.w2_weight
    assert arguments["shared_gate_up_weight"] is shared.gate_up_proj.weight
    assert arguments["shared_down_weight"] is shared.down_proj.weight
    assert (arguments["topk"], arguments["routed_scaling_factor"]) == (8, 2.5)
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK


def unquantized_runner(layer):
    return layer.mlp.experts.quant_method._aiter_runner


DRAFT_LAYER_FALLBACKS = {
    "expert-parallel": lambda s, layer, batch: setattr(layer.mlp, "moe_ep_size", 2),
    "no-aiter-runner": lambda s, layer, batch: setattr(layer.mlp.experts.quant_method, "_aiter_runner", None),
    "flashinfer": lambda s, layer, batch: setattr(
        layer.mlp.experts.quant_method, "use_flashinfer_cutlass", True
    ),
    "gelu": lambda s, layer, batch: setattr(unquantized_runner(layer).config, "activation", "gelu"),
    "weights-on-input": lambda s, layer, batch: setattr(
        unquantized_runner(layer).config, "apply_router_weight_on_input", True
    ),
    "no-combine": lambda s, layer, batch: setattr(unquantized_runner(layer).config, "no_combine", True),
    "expert-bias": lambda s, layer, batch: setattr(layer.mlp.experts, "w13_weight_bias", torch.zeros(9, 64)),
    "expert-mask": lambda s, layer, batch: setattr(
        layer.mlp.experts.dispatcher, "expert_mask_gpu", torch.ones(9)
    ),
    "unshuffled": lambda s, layer, batch: setattr(layer.mlp.experts, "_skip_aiter_moe_shuffle", True),
    "hash-routing": lambda s, layer, batch: setattr(layer.mlp, "is_hash", True),
    "shared-expert-apart": lambda s, layer, batch: setattr(layer.mlp, "num_fused_shared_experts", 0),
    "two-fused-shared-experts": lambda s, layer, batch: setattr(layer.mlp, "num_fused_shared_experts", 2),
}


@pytest.mark.parametrize("change", DRAFT_LAYER_FALLBACKS.values(), ids=list(DRAFT_LAYER_FALLBACKS))
def test_draft_layer_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = draft_decoder_layer(), draft_extend(sglang)
    change(sglang, layer, forward_batch)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


DRAFT_LAYER_SHARED_APART_FALLBACKS = {
    "no-shared-expert": lambda layer: setattr(layer.mlp, "shared_experts", None),
    "shared-expert-replicated": lambda layer: setattr(layer.mlp, "_shared_expert_tp1", True),
    "fp8-shared-expert": lambda layer: setattr(
        layer.mlp.shared_experts.down_proj, "weight", torch.zeros(256, 32, dtype=FP8)
    ),
    "shared-swiglu-limit": lambda layer: setattr(layer.mlp.shared_experts, "swiglu_limit", 7.0),
}


@pytest.mark.parametrize(
    "change",
    DRAFT_LAYER_SHARED_APART_FALLBACKS.values(),
    ids=list(DRAFT_LAYER_SHARED_APART_FALLBACKS),
)
def test_draft_layer_with_the_shared_expert_apart_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = draft_decoder_layer_shared_apart(), draft_extend(sglang)
    change(layer)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def fp8_decoder_layer():
    """decoder_layer() of GLM-5.3-FP8: its MoE's experts on Fp8MoEMethod's aiter runner, the
    8 routed and the shared one fused last, fp8 at a scale per 128 x 128 block and shuffled
    as sglang stores them."""
    layer = decoder_layer()
    method = type("Fp8MoEMethod", (SimpleNamespace,), {})(
        runner=SimpleNamespace(
            runner_backend=SimpleNamespace(is_aiter=lambda: True),
            config=runner().config,
        ),
        block_quant=True,
        use_mxfp8=False,
        is_fp4_expert=False,
        weight_block_size=[128, 128],
    )
    w13, w2 = torch.zeros(9, 512, 256, dtype=FP8), torch.zeros(9, 256, 256, dtype=FP8)
    w13.is_shuffled = w2.is_shuffled = True
    layer.mlp.experts = SimpleNamespace(
        quant_method=method,
        w13_weight=w13,
        w13_weight_scale_inv=torch.zeros(9, 4, 2),
        w2_weight=w2,
        w2_weight_scale_inv=torch.zeros(9, 2, 2),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
    )
    return layer


def test_fp8_layer_runs_its_moe_op(sglang, fake_ops):
    layer = fp8_decoder_layer()
    _, (out, topk) = run_layer(layer, target_verify(sglang), prev_topk_indices=TOPK)
    (attention, _), (op, arguments) = fake_ops
    assert (attention, op) == ("deepseek_v2_attention", "deepseek_v2_fp8_moe_layer")
    point = contract_point(arguments, DEEPSEEK_V2_FP8_MOE_LAYER)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    experts = layer.mlp.experts
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w1_scale"] is experts.w13_weight_scale_inv
    assert arguments["w2"] is experts.w2_weight
    assert arguments["w2_scale"] is experts.w2_weight_scale_inv
    assert (arguments["topk"], arguments["routed_scaling_factor"]) == (8, 2.5)
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK


def fp8_method(layer):
    return layer.mlp.experts.quant_method


FP8_LAYER_FALLBACKS = {
    "shared-expert-apart": lambda layer: setattr(layer.mlp, "num_fused_shared_experts", 0),
    "per-channel-scales": lambda layer: setattr(fp8_method(layer), "block_quant", False),
    "mxfp8": lambda layer: setattr(fp8_method(layer), "use_mxfp8", True),
    "fp4-experts": lambda layer: setattr(fp8_method(layer), "is_fp4_expert", True),
    "other-blocks": lambda layer: setattr(fp8_method(layer), "weight_block_size", [128, 64]),
    "no-runner": lambda layer: setattr(fp8_method(layer), "runner", None),
    "triton-runner": lambda layer: setattr(
        fp8_method(layer).runner.runner_backend, "is_aiter", lambda: False
    ),
    "gelu": lambda layer: setattr(fp8_method(layer).runner.config, "activation", "gelu"),
    "expert-mask": lambda layer: setattr(layer.mlp.experts.dispatcher, "expert_mask_gpu", torch.ones(9)),
}


@pytest.mark.parametrize("change", FP8_LAYER_FALLBACKS.values(), ids=list(FP8_LAYER_FALLBACKS))
def test_fp8_layer_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = fp8_decoder_layer(), target_verify(sglang)
    change(layer)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def fp8_draft_decoder_layer():
    """fp8_decoder_layer() as GLM-5.3-FP8's MTP draft layer: the 8 routed experts alone, and
    the shared expert apart, an fp8 DeepseekV2MLP of intermediate size 256 with block scales
    (sglang fuses no shared expert into a draft's experts)."""
    layer = fp8_decoder_layer()
    mlp, routing = layer.mlp, layer.mlp.topk.topk_config
    w13, w2 = torch.zeros(8, 512, 256, dtype=FP8), torch.zeros(8, 256, 256, dtype=FP8)
    w13.is_shuffled = w2.is_shuffled = True
    mlp.experts.w13_weight, mlp.experts.w2_weight = w13, w2
    mlp.experts.w13_weight_scale_inv = torch.zeros(8, 4, 2)
    mlp.experts.w2_weight_scale_inv = torch.zeros(8, 2, 2)
    mlp.num_fused_shared_experts = routing.num_fused_shared_experts = 0
    routing.top_k = 8
    mlp.shared_experts = SimpleNamespace(
        gate_up_proj=linear(512, 256), down_proj=linear(256, 256), swiglu_limit=None
    )
    return layer


def test_fp8_draft_layer_runs_its_moe_op_with_the_shared_expert_apart(sglang, fake_ops):
    layer = fp8_draft_decoder_layer()
    _, (out, topk) = run_layer(layer, draft_extend(sglang), prev_topk_indices=TOPK)
    (attention, _), (op, arguments) = fake_ops
    assert (attention, op) == (
        "deepseek_v2_attention",
        "deepseek_v2_fp8_moe_layer_shared_apart",
    )
    point = contract_point(arguments, DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    experts, shared = layer.mlp.experts, layer.mlp.shared_experts
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w2_scale"] is experts.w2_weight_scale_inv
    assert arguments["shared_gate_up_weight"] is shared.gate_up_proj.weight
    assert arguments["shared_gate_up_scale"] is shared.gate_up_proj.weight_scale_inv
    assert arguments["shared_down_weight"] is shared.down_proj.weight
    assert arguments["shared_down_scale"] is shared.down_proj.weight_scale_inv
    assert (arguments["topk"], arguments["routed_scaling_factor"]) == (8, 2.5)
    assert torch.equal(out, arguments["hidden_states"] + 1) and topk is TOPK


FP8_DRAFT_LAYER_FALLBACKS = {
    "no-shared-expert": lambda layer: setattr(layer.mlp, "shared_experts", None),
    "shared-expert-replicated": lambda layer: setattr(layer.mlp, "_shared_expert_tp1", True),
    "bf16-shared-down": lambda layer: setattr(
        layer.mlp.shared_experts.down_proj,
        "weight",
        torch.zeros(256, 256, dtype=torch.bfloat16),
    ),
    "shared-swiglu-limit": lambda layer: setattr(layer.mlp.shared_experts, "swiglu_limit", 7.0),
}


@pytest.mark.parametrize("change", FP8_DRAFT_LAYER_FALLBACKS.values(), ids=list(FP8_DRAFT_LAYER_FALLBACKS))
def test_fp8_draft_layer_falls_back_to_sglang(sglang, fake_ops, change):
    layer, forward_batch = fp8_draft_decoder_layer(), draft_extend(sglang)
    change(layer)
    _, result = run_layer(layer, forward_batch, prev_topk_indices=TOPK)
    assert isinstance(result, FallbackReason) and fake_ops == []


def nextn_model(sglang_layer=False):
    """sglang's DeepseekModelNextN, shrunk to hidden size 256: embeddings of 1s; its head norm
    triples, after adding the residual when given one. Its decoder, a layer of the route's
    own, writes the stream plus 1 back as the residual; as a layer of sglang's own
    (``sglang_layer``), it writes its input as the residual and leaves its MoE's output of 1s
    pending, the sum over 8 ranks still owed. Either carries the top-k on plus 2."""
    embeds = torch.ones(4, 256, dtype=torch.bfloat16)

    def decoder(positions, hidden_states, forward_batch, allocator, prev_topk_indices):
        stream = forward_batch.residual_stream
        model.decoder_calls.append((hidden_states, stream.residual, stream.pending, prev_topk_indices))
        if not sglang_layer:
            return stream.write(hidden_states + 1), prev_topk_indices + 2
        stream.write(hidden_states)
        group = SimpleNamespace(all_reduce=lambda partial: partial * 8)
        output = boundary("output").UnreducedOutput(torch.ones_like(embeds), group)
        return stream.record(output, PLAIN_ADD), prev_topk_indices + 2

    def head_norm(x, residual=None):
        if residual is None:
            return x * 3
        return (x + residual) * 3, x + residual

    model = SimpleNamespace(
        embeds=embeds,
        embed_input_ids=lambda input_ids, forward_batch: embeds,
        enorm=norm(),
        hnorm=norm(),
        eh_proj=SimpleNamespace(weight=torch.zeros(256, 512, dtype=torch.bfloat16)),
        rot_weight=None,
        decoder=decoder,
        decoder_calls=[],
        shared_head=SimpleNamespace(norm=head_norm),
    )
    return model


def draft_batch():
    """A draft batch of 4 tokens: the target's hidden states 2s, the carried top-k TOPK, and
    no residual stream started."""
    return SimpleNamespace(
        mm_input_embeds=None,
        spec_info=SimpleNamespace(hidden_states=torch.full((4, 256), 2.0, dtype=torch.bfloat16)),
        carried_topk=TOPK,
        residual_stream=None,
    )


def run_nextn(model, batch, **kwargs):
    return deepseek_v2._model_nextn(model, torch.arange(4), torch.arange(4), batch, **kwargs)


def test_nextn_takes_its_decoder_input_from_its_op(sglang, fake_ops):
    model, batch = nextn_model(), draft_batch()
    out = run_nextn(model, batch)
    ((op, arguments),) = fake_ops
    assert op == "deepseek_v2_eh_proj"
    assert contract_point(arguments, DEEPSEEK_V2_EH_PROJ) is not None
    assert arguments["embeds"] is model.embeds
    assert arguments["hidden"] is batch.spec_info.hidden_states
    # the decoder's input starts a stream of its own
    ((hidden_states, residual, pending, prev_topk),) = model.decoder_calls
    assert (hidden_states == 5).all() and residual is None and pending is None
    assert prev_topk is TOPK
    # the head norm on what the decoder leaves, the stream closed, the top-k carried on
    assert (out == 18).all() and batch.residual_stream is None
    (share,) = sglang.topk_shares
    assert torch.equal(share.updated, TOPK + 2) and share.published


def test_nextn_norms_what_a_layer_of_sglangs_own_leaves(sglang, fake_ops):
    """A decoder layer that fell back to sglang: the head norm takes its output's sum over
    the ranks added to the residual."""
    model, batch = nextn_model(sglang_layer=True), draft_batch()
    out = run_nextn(model, batch)
    assert (out == (1 * 8 + 5) * 3).all() and batch.residual_stream is None
    (share,) = sglang.topk_shares
    assert torch.equal(share.updated, TOPK + 2) and share.published


# each: change(sglang, model, batch, kwargs) that makes the route fall back
NEXTN_FALLBACKS = {
    "embeddings-in": lambda s, model, batch, kwargs: kwargs.update(input_embeds=torch.zeros(4, 256)),
    "multimodal": lambda s, model, batch, kwargs: setattr(batch, "mm_input_embeds", torch.zeros(4, 256)),
    "rotated": lambda s, model, batch, kwargs: setattr(model, "rot_weight", torch.zeros(256, 256)),
    "two-eps": lambda s, model, batch, kwargs: setattr(model.hnorm, "variance_epsilon", 1e-6),
    # each rank then drafts its own requests, while the op sums the ranks' shares of one set
    # of tokens
    "dp-attention": lambda s, model, batch, kwargs: setattr(s, "dp_attention", True),
}


@pytest.mark.parametrize("change", NEXTN_FALLBACKS.values(), ids=list(NEXTN_FALLBACKS))
def test_nextn_falls_back_to_sglang(sglang, fake_ops, change):
    model, batch, kwargs = nextn_model(), draft_batch(), {}
    change(sglang, model, batch, kwargs)
    result = run_nextn(model, batch, **kwargs)
    assert isinstance(result, FallbackReason) and fake_ops == []
    assert batch.residual_stream is None


def test_nextn_returns_its_op_fallback(sglang, monkeypatch):
    """The op's reason, before the residual stream starts, the decoder runs or the top-k is
    shared."""
    monkeypatch.setattr(
        deepseek_v2,
        "deepseek_v2_eh_proj",
        lambda **arguments: FallbackReason.NO_WORKLOAD,
    )
    model, batch = nextn_model(), draft_batch()
    assert run_nextn(model, batch) is FallbackReason.NO_WORKLOAD
    assert batch.residual_stream is None
    assert model.decoder_calls == [] and sglang.topk_shares == []


def forward_normal_moe():
    """moe_layer() as DeepseekV2MoE.forward_normal reads it: its output summed over the
    ranks at its end (reduce_results), the shared expert not run beside the combine."""
    mlp = moe_layer()
    mlp.reduce_results = True
    mlp._fuse_shared_experts_inside_sbo = False
    return mlp


def run_moe(mlp, hidden=None, **kwargs):
    """The forward_normal route on hidden (default 2s), and its result."""
    if hidden is None:
        hidden = torch.full((4, 256), 2.0, dtype=torch.bfloat16)
    return hidden, deepseek_v2._moe_forward_normal(mlp, hidden, None, None, **kwargs)


def test_moe_forward_normal_reaches_its_op(sglang, fake_ops):
    """The router, the top-k and the experts, the shared one fused last, as one op call on
    the MoE's input; sglang's post-experts all-reduce then sums the op's output."""
    mlp = forward_normal_moe()
    hidden, out = run_moe(mlp)
    ((op, arguments),) = fake_ops
    assert op == "deepseek_v2_moe"
    point = contract_point(arguments, DEEPSEEK_V2_MOE)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    experts, routing = mlp.experts, mlp.topk.topk_config
    assert arguments["hidden_states"] is hidden
    assert arguments["router_weight"] is mlp.gate.weight
    assert arguments["correction_bias"] is routing.correction_bias
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w1_scale"] is experts.w13_weight_scale
    assert arguments["w2"] is experts.w2_weight
    assert arguments["w2_scale"] is experts.w2_weight_scale
    assert (arguments["topk"], arguments["routed_scaling_factor"]) == (8, 2.5)
    (summed,) = sglang.all_reduced
    assert torch.equal(summed, hidden + 1) and torch.equal(out, (hidden + 1) * 8)


def test_moe_forward_normal_leaves_the_sum_to_a_later_step(sglang, fake_ops):
    """Without reduce_results a later step (the layer's boundary) sums the output: the
    route returns the op's."""
    mlp = forward_normal_moe()
    mlp.reduce_results = False
    hidden, out = run_moe(mlp)
    assert torch.equal(out, hidden + 1) and sglang.all_reduced == []


def test_moe_forward_normal_returns_its_op_fallback(sglang, monkeypatch):
    monkeypatch.setattr(deepseek_v2, "deepseek_v2_moe", lambda **_: FallbackReason.NO_WORKLOAD)
    _, result = run_moe(forward_normal_moe())
    assert result is FallbackReason.NO_WORKLOAD and sglang.all_reduced == []


# each: forward_normal's arguments that make its route fall back
FORWARD_NORMAL_ARGUMENT_FALLBACKS = {
    "shared-expert-skipped": {"skip_shared_experts": True},
    "padded-rows-masked": {"num_token_non_padded": torch.tensor(3)},
    "vision-topk": {"use_vision_topk": True},
    "output-in-pieces": {"return_moe_output": True},
    "output-sum-left": {"leave_output_sum": True},
}


@pytest.mark.parametrize(
    "kwargs",
    FORWARD_NORMAL_ARGUMENT_FALLBACKS.values(),
    ids=list(FORWARD_NORMAL_ARGUMENT_FALLBACKS),
)
def test_moe_forward_normal_falls_back_on_its_arguments(sglang, fake_ops, kwargs):
    _, result = run_moe(forward_normal_moe(), **kwargs)
    assert isinstance(result, FallbackReason)
    assert fake_ops == [] and sglang.all_reduced == []


def mxfp4_scheme_runner(s, mlp):
    return mlp.experts.scheme.runner


def experts_lora(s, mlp):
    # sglang's FusedMoEWithLoRA copies quant_method, not scheme.
    del mlp.experts.scheme
    mlp.experts.quant_method = type("QuarkFusedMoEMethod", (), {})()


# each: change(sglang, mlp) that makes forward_normal's route fall back
FORWARD_NORMAL_FALLBACKS = {
    "shared-expert-beside-combine": lambda s, mlp: setattr(mlp, "_fuse_shared_experts_inside_sbo", True),
    "hash-routing": lambda s, mlp: setattr(mlp, "is_hash", True),
    "vision-bias": lambda s, mlp: setattr(mlp.gate, "e_score_correction_bias_vl", torch.zeros(8)),
    "deterministic": lambda s, mlp: setattr(s.config.deterministic, "enable_deterministic_inference", True),
    "eplb": lambda s, mlp: setattr(s.config.moe, "enable_eplb", True),
    "recorder": lambda s, mlp: setattr(s.config.moe, "expert_distribution_recorder_mode", "stat"),
    "return-routed-experts": lambda s, mlp: setattr(s.config.features, "enable_return_routed_experts", True),
    "expert-parallel": lambda s, mlp: setattr(mlp, "moe_ep_size", 2),
    "shared-expert-apart": lambda s, mlp: setattr(mlp, "num_fused_shared_experts", 0),
    "no-aiter-runner": lambda s, mlp: setattr(mlp.experts.scheme, "runner", None),
    "gelu": lambda s, mlp: setattr(mxfp4_scheme_runner(s, mlp).config, "activation", "gelu"),
    "expert-mask": lambda s, mlp: setattr(mlp.experts.dispatcher, "expert_mask_gpu", torch.ones(8)),
    "experts-lora": experts_lora,
    # experts the route has no op for yet
    "fp8-experts": lambda s, mlp: setattr(mlp, "experts", fp8_decoder_layer().mlp.experts),
    "unquantized-experts": lambda s, mlp: setattr(mlp, "experts", unquantized_experts()),
}


@pytest.mark.parametrize("change", FORWARD_NORMAL_FALLBACKS.values(), ids=list(FORWARD_NORMAL_FALLBACKS))
def test_moe_forward_normal_falls_back_to_sglang(sglang, fake_ops, change):
    mlp = forward_normal_moe()
    change(sglang, mlp)
    _, result = run_moe(mlp)
    assert isinstance(result, FallbackReason)
    assert fake_ops == [] and sglang.all_reduced == []


def test_draft_keeps_its_logits_at_topk_1(sglang):
    logits = torch.zeros(1, 1024)
    assert eagle._renorm_draft_probs(logits, None, False) is logits
    # rejection sampling and more tokens a step take the softmax
    assert isinstance(eagle._renorm_draft_probs(logits, None, True), FallbackReason)
    sglang.spec.speculative_eagle_topk = 2
    assert isinstance(eagle._renorm_draft_probs(logits, None, False), FallbackReason)


def test_draft_top1_reaches_its_op_at_a_point_of_its_contract(sglang, fake_ops):
    logits = torch.zeros(1, 1024)
    assert eagle._fast_topk(logits, 1, -1) == "out"
    ((op, arguments),) = fake_ops
    assert op == "eagle_worker_v2_draft_top1"
    assert contract_point(arguments, EAGLE_WORKER_V2_DRAFT_TOP1) is not None
    assert isinstance(eagle._fast_topk(logits, 2, -1), FallbackReason)
    assert len(fake_ops) == 1


def test_draft_steps_take_the_draft_top1_and_advance_their_positions(sglang, fake_ops):
    logits, positions = torch.zeros(4, 1024), torch.arange(4)
    assert eagle._draft_topk1_postprocess(logits, positions, None, 1) == "out"
    ((op, arguments),) = fake_ops
    assert op == "eagle_worker_v2_draft_top1" and arguments["logits"] is logits
    assert contract_point(arguments, EAGLE_WORKER_V2_DRAFT_TOP1) is not None
    assert torch.equal(positions, torch.arange(1, 5))


def test_draft_steps_fall_back_with_their_positions_as_they_were(sglang, fake_ops, monkeypatch):
    """With draft tokens to write the argmaxes into (sglang passes them on CUDA), or when the
    op falls back."""
    logits, positions = torch.zeros(4, 1024), torch.arange(4)
    draft_tokens = torch.zeros(4, 3, dtype=torch.int64)
    result = eagle._draft_topk1_postprocess(logits, positions, draft_tokens, 1)
    assert isinstance(result, FallbackReason) and fake_ops == []
    monkeypatch.setattr(eagle, "eagle_worker_v2_draft_top1", lambda logits: FallbackReason.NO_WORKLOAD)
    result = eagle._draft_topk1_postprocess(logits, positions)
    assert result is FallbackReason.NO_WORKLOAD
    assert torch.equal(positions, torch.arange(4))


ROUTES = {
    "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward": qwen3_5._gdn_forward,
    "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts": (
        qwen2_moe._forward_router_experts
    ),
    "sglang.srt.layers.layernorm.RMSNorm.forward_aiter": norms._rms_norm_forward_aiter,
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend."
    "_forward_triton_decode": dsa_backend._forward_triton_decode,
    "sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.forward": (deepseek_v2._decoder_layer),
    "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal": (deepseek_v2._moe_forward_normal),
    "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.forward": (deepseek_v2._model_nextn),
    "sglang.srt.models.minimax_m2.MiniMaxM2MoE.forward_normal": (minimax_m2._moe_forward_normal),
    "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs": (eagle._renorm_draft_probs),
    "sglang.srt.speculative.eagle_worker_v2.fast_topk": eagle._fast_topk,
    "sglang.srt.speculative.eagle_worker_v2.draft_topk1_postprocess": (eagle._draft_topk1_postprocess),
}


@pytest.mark.parametrize("release", ["0.5.20", "0.5.21", "0.5.22", "0.5.23"])
def test_each_target_has_one_route_on_0_5_22_and_none_on_another_sglang(release):
    """These op packages' routes; a model backend's (deepseek_v4) are its own tests'."""
    packages = {
        m.__name__
        for m in (
            qwen3_5,
            qwen2_moe,
            norms,
            dsa_backend,
            deepseek_v2,
            minimax_m2,
            eagle,
        )
    }
    routes = {}
    for target, versions, route, _ in hook._ROUTES:
        if route.__module__ in packages and versions.contains(release):
            routes.setdefault(target, []).append(route)
    expected = {target: [route] for target, route in ROUTES.items()}
    assert routes == (expected if release == "0.5.22" else {})
