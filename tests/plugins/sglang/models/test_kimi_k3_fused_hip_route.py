# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on sglang 0.5.24.dev809's KimiK3MoEHipMixin._forward_fused_hip (Kimi-K3.1-Pro's image), on stand-ins of
the layer and of the sglang modules it reads: which batches reach kimi_k3_latent_moe, with its fp32 front, and which
of sglang's own tails runs after the op; the op is a recorder."""

import contextlib
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import hook
from aklr.plugins.sglang.models.kimi_k3 import api as kimi_k3

TOKENS, HIDDEN, LATENT, EXPERTS, INTER, SHARED = 6, 256, 128, 32, 128, 128
HIP_TARGET = "sglang.srt.models.kimi_k3_amd.moe.KimiK3MoEHipMixin._forward_fused_hip"


def _named(name, **attrs):
    """An object whose type is called ``name``, as the route's checks read it."""
    return type(name, (), attrs)()


@pytest.fixture
def sglang(monkeypatch):
    """The sglang 0.5.24 modules the route reads, recording what of them runs."""
    ran = []

    def module(name, **attrs):
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    @contextlib.contextmanager
    def symmetric(group, disabled):
        ran.append("symmetric")
        yield

    tail = SimpleNamespace(
        supports_latent_tail=lambda latent, weight, shared, prefix: True,
        k3_latent_tail=lambda *args: ran.append("k3_latent_tail") or "fused tail",
    )
    hip_moe = module(
        "sglang.srt.models.kimi_k3_amd.moe",
        _UP_PROJ_COLUMN_SPLIT_MIN_TOKENS=2048,
        use_symmetric_memory=symmetric,
        get_parallel=lambda: SimpleNamespace(tp_group=None),
        is_allocation_symmetric=lambda: True,
        kimi_k3_latent_tail_hip=tail,
    )
    module("sglang.srt.models.kimi_k3_amd", moe=hip_moe)
    model = module(
        "sglang.srt.models.kimi_k3", _add3=lambda *args, prefetch_bc=False: ran.append("add3") or "added"
    )
    module("sglang.srt.models", kimi_k3=model)
    communication = module("sglang.srt.layers.communication")
    communication.k3_ar_fusion = module("sglang.srt.layers.communication.k3_ar_fusion", enabled=lambda: False)
    quantization = module("sglang.srt.layers.quantization")
    quantization.mxfp4 = module(
        "sglang.srt.layers.quantization.mxfp4", _aiter_situ_uses_gu_interleaved_weights=lambda: True
    )
    module("sglang.srt.runtime_context", get_parallel=lambda: SimpleNamespace(moe_ep_size=1))
    monkeypatch.setattr(kimi_k3.moe_blocks, "runtime_falls_back", lambda: False)
    return SimpleNamespace(ran=ran, tail=tail)


@pytest.fixture
def op(monkeypatch):
    """The op's calls, answered with ``answer[0]`` (its outputs by default); ``reason[0]`` is what its
    fallback_reason answers."""
    calls, answer, reason = [], [None], [None]

    def record(**call):
        calls.append(call)
        return answer[0] if answer[0] is not None else (call["latent"], call["shared"])

    record.fallback_reason = lambda **call: reason[0]
    monkeypatch.setattr(kimi_k3, "kimi_k3_latent_moe", record)
    return SimpleNamespace(calls=calls, answer=answer, reason=reason)


def k31_moe(sglang, **changes):
    """KimiK3MoE shrunk, as sglang 0.5.24 holds it on aiter after load: the fp32 merged front, MXFP4 experts in
    aiter's gate-up-interleaved preshuffle on its runner, SiTU 4 / 25 shared experts, the latent norm and up
    projection, TopKConfig's shared-expert layout."""
    experts = _named(
        "FusedMoE",
        w13_weight=torch.zeros(EXPERTS, 2 * INTER, LATENT // 2, dtype=torch.uint8),
        w2_weight=torch.zeros(EXPERTS, LATENT, INTER // 2, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(EXPERTS * 2 * INTER, LATENT // 32, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(EXPERTS * LATENT, 8, dtype=torch.uint8),
        num_experts=EXPERTS,
        reduce_results=False,
        quant_method=_named(
            "Mxfp4MoEMethod",
            with_bias=False,
            hidden_pad=0,
            intermediate_pad=0,
            runner=SimpleNamespace(runner_backend=SimpleNamespace(is_aiter=lambda: True)),
        ),
        dispatcher=_named("StandardDispatcher"),
        moe_runner_config=SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            gate_up_interleaved=False,
            apply_router_weight_on_input=False,
            no_combine=False,
        ),
    )
    bias = torch.zeros(EXPERTS)
    config = SimpleNamespace(
        top_k=16,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        custom_routing_function=None,
        scoring_func="sigmoid",
        correction_bias=bias,
        renormalize=True,
        shared_experts=SimpleNamespace(num_fused_shared_experts=0),
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    moe = SimpleNamespace(
        experts=experts,
        shared_experts=SimpleNamespace(
            down_proj=SimpleNamespace(weight=torch.zeros(HIDDEN, SHARED, dtype=torch.bfloat16)),
            act_fn=_named("SituAndMul", beta=4.0, linear_beta=25.0),
        ),
        num_shared_experts=1,
        moe_hidden_size=LATENT,
        _eligible_for_fused_front=True,
        _front_fp32=True,
        _front_w=torch.zeros(2 * SHARED + EXPERTS + LATENT, HIDDEN, dtype=torch.bfloat16),
        _front_sizes=[2 * SHARED, EXPERTS, LATENT],
        gate=SimpleNamespace(e_score_correction_bias=bias),
        topk=SimpleNamespace(topk_config=config, waterfill_balancer=None, enable_waterfill=False),
        routed_expert_norm=SimpleNamespace(weight=torch.ones(LATENT), variance_epsilon=1e-6),
        routed_expert_up_proj=_UpProj(torch.zeros(HIDDEN, LATENT, dtype=torch.bfloat16), sglang.ran),
        _a8w4_moe_min_tokens=2048,
        _prefill_up_proj_cols=slice(0, HIDDEN // 8),
        next_attn_read=None,
        _fused_latent_tail_eligible=False,
        _row_sharded_tail_eligible=lambda n, has_prefix: n >= 4096 and has_prefix,
        _reduce_latent_and_shared=lambda buf: sglang.ran.append("all_reduce") or buf,
        _column_tail_and_next_read=lambda latent, shared, prefix: (
            sglang.ran.append("next_read") or "next read"
        ),
        _latent_norm=lambda latent: sglang.ran.append("norm") or latent,
    )
    for name, value in changes.items():
        setattr(moe, name, value)
    return moe


class _UpProj:
    """routed_expert_up_proj: a ReplicatedLinear's (output, bias) call, recorded."""

    def __init__(self, weight, ran):
        self.weight, self.quant_method, self._ran = weight, _named("UnquantizedLinearMethod"), ran

    def __call__(self, x):
        self._ran.append("up_proj")
        return x.new_zeros(x.shape[0], self.weight.shape[0]), None


def forward(moe, tokens=TOKENS, prefix=True):
    hidden = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16)
    prefix_sum = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16) if prefix else None
    return kimi_k3._forward_fused_hip(moe, hidden, prefix_sum=prefix_sum)


def test_the_route_is_0_5_24s_rocm_path_and_takes_its_parameters():
    """KimiK3MoE._forward_fused hands every call to the mixin's _forward_fused_hip on ROCm (0.5.24): the route takes
    that one, on the model image only; 0.5.22's and 0.5.23's _forward_fused keeps its own route."""
    routes = {target: (versions, route) for target, versions, route in hook._ROUTES}
    versions, route = routes[HIP_TARGET]
    assert versions == ("0.5.24.dev809",) and route is kimi_k3._forward_fused_hip
    assert routes["sglang.srt.models.kimi_k3.KimiK3MoE._forward_fused"][0] == ("0.5.22", "0.5.23")
    # sglang 0.5.24: def _forward_fused_hip(self, hidden_states, *, prefix_sum)
    assert [(p.name, p.kind) for p in inspect.signature(route).parameters.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("hidden_states", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("prefix_sum", inspect.Parameter.KEYWORD_ONLY),
    ]


def test_a_decode_batch_reaches_the_op_with_the_fp32_front(sglang, op):
    """The op writes both partial sums into one symmetric buffer, latent then shared, as sglang lays it out; then
    sglang's all-reduce, latent norm, up projection and tail add."""
    moe = k31_moe(sglang)
    assert forward(moe) == "added"
    (call,) = op.calls
    assert call["front_fp32"] is True and call["topk"] == 16
    assert (
        call["front_weight"] is moe._front_w and call["correction_bias"] is moe.gate.e_score_correction_bias
    )
    assert call["w1"] is moe.experts.w13_weight and call["w2_scale"] is moe.experts.w2_weight_scale
    assert call["shared_w2"] is moe.shared_experts.down_proj.weight
    latent, shared = call["latent"], call["shared"]
    assert latent.shape == (TOKENS, LATENT) and shared.shape == (TOKENS, HIDDEN)
    assert shared.data_ptr() == latent.data_ptr() + TOKENS * LATENT * latent.element_size()
    assert sglang.ran == ["symmetric", "all_reduce", "norm", "up_proj", "add3"]


def test_the_next_layers_attention_read_rides_the_tail_as_on_sglang(sglang, op):
    """With the next layer's read linked (link_column_tails_hip), sglang's column-parallel tail runs it."""
    moe = k31_moe(sglang, next_attn_read=object(), _fused_latent_tail_eligible=True)
    assert forward(moe) == "next read"
    assert sglang.ran == ["symmetric", "all_reduce", "next_read"]


def test_the_fused_latent_tail_runs_where_sglang_runs_it(sglang, op):
    moe = k31_moe(sglang, _fused_latent_tail_eligible=True)
    assert forward(moe) == "fused tail"
    assert sglang.ran == ["symmetric", "all_reduce", "k3_latent_tail"]


@pytest.mark.parametrize(
    "tokens", [2048, 3000, 4096], ids=["a8w4_experts", "column_split_tail", "row_sharded_tail"]
)
def test_a_prefill_batch_of_sglangs_own_experts_or_tails_stays_on_sglang(sglang, op, tokens):
    assert forward(k31_moe(sglang), tokens=tokens) is FallbackReason.OUT_OF_CONTRACT
    assert op.calls == [] and sglang.ran == []


@pytest.mark.parametrize(
    "changes",
    [
        {"_front_fp32": False},
        {"_eligible_for_fused_front": False},
        {"num_shared_experts": None},
    ],
    ids=["bf16_front", "not_fused_front", "no_shared_experts"],
)
def test_a_layer_off_the_contract_stays_on_sglang(sglang, op, changes):
    assert forward(k31_moe(sglang, **changes)) is FallbackReason.OUT_OF_CONTRACT
    assert op.calls == []


def test_a_call_the_op_does_not_serve_allocates_nothing(sglang, op):
    """Asked before the buffer: a decode batch no kernel serves leaves sglang's own path untouched."""
    op.reason[0] = FallbackReason.NO_WORKLOAD
    assert forward(k31_moe(sglang)) is FallbackReason.NO_WORKLOAD
    assert op.calls == [] and sglang.ran == []


def test_a_call_the_op_declines_after_asking_goes_back_to_sglang(sglang, op):
    op.answer[0] = FallbackReason.NOT_WARMED
    assert forward(k31_moe(sglang)) is FallbackReason.NOT_WARMED
    assert sglang.ran == ["symmetric"]


@pytest.mark.parametrize("fused", [0, 1])
def test_the_router_reads_the_fused_shared_experts_of_either_topk_config(fused):
    """0.5.22's TopKConfig counts the fused shared experts itself; 0.5.23's and 0.5.24's in its shared_experts."""
    common = dict(
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        custom_routing_function=None,
        scoring_func="sigmoid",
        correction_bias=torch.zeros(4),
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    old = SimpleNamespace(**common, num_fused_shared_experts=fused)
    new = SimpleNamespace(**common, shared_experts=SimpleNamespace(num_fused_shared_experts=fused))
    for config in (old, new):
        topk = SimpleNamespace(topk_config=config, waterfill_balancer=None, enable_waterfill=False)
        assert kimi_k3._topk_falls_back(topk) is bool(fused)
