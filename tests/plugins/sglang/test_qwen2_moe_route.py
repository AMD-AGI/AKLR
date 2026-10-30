# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the Qwen2MoeSparseMoeBlock route tells its ops, and when it falls back to sglang."""

import sys
from types import SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.models.qwen2_moe.api as qwen2_moe
from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.qwen2_moe.contract import (
    QWEN2_MOE_FP8_EXPERTS,
    QWEN2_MOE_FP8_ROUTER_EXPERTS,
    QWEN2_MOE_ROUTER,
)

FP8 = torch.float8_e4m3fn
OPS = ("qwen2_moe_fp8_router_experts", "qwen2_moe_router", "qwen2_moe_fp8_experts")


def zeros(*shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype)


def contract_point(arguments, contract):
    """The Point of the call the route made, as dispatch keys it; None outside the contract."""
    codec = FastKeyCodec.from_contract(contract)
    call = {p.name: p.default for p in contract.api_params} | arguments
    key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()(call)
    return None if key is None else codec.to_point(key)


class Experts:
    """sglang's FusedMoE of Qwen3.5's MoE block on one rank on aiter's runner, shrunk: 16
    routed experts and the shared one last, hidden and intermediate size 256, fp8 at a
    scale per 128 x 128 block, shuffled for aiter."""

    def __init__(self):
        fp8 = sys.modules["sglang.srt.layers.quantization.fp8"]
        runner = SimpleNamespace(
            runner_backend=SimpleNamespace(is_aiter=lambda: True),
            config=SimpleNamespace(
                activation="silu",
                is_gated=True,
                apply_router_weight_on_input=False,
                no_combine=False,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
            ),
        )
        self.quant_method = fp8.Fp8MoEMethod(
            block_quant=True,
            use_mxfp8=False,
            is_fp4_expert=False,
            weight_block_size=[128, 128],
            runner=runner,
        )
        self.w13_weight = zeros(17, 512, 256, dtype=FP8)
        self.w2_weight = zeros(17, 256, 256, dtype=FP8)
        self.w13_weight.is_shuffled = self.w2_weight.is_shuffled = True
        self.w13_weight_scale_inv = zeros(17, 4, 2, dtype=torch.float32)
        self.w2_weight_scale_inv = zeros(17, 2, 2, dtype=torch.float32)
        self.dispatcher = SimpleNamespace(expert_mask_gpu=None)
        self.reduce_results = False
        self._dwdp_bound = False
        self.calls = []

    def __call__(self, hidden_states, topk_output):
        self.calls.append((hidden_states, topk_output))
        return "sglang experts"


def block():
    """sglang's Qwen2MoeSparseMoeBlock of Qwen3.5 with the shared expert fused into the
    experts, as aiter's path builds it; _append_shared_to_topk_output records its call.
    """
    appended = []

    def append(topk_output, hidden_states):
        appended.append((topk_output, hidden_states))
        return "with the shared expert"

    return SimpleNamespace(
        enable_shared_expert_fusion=True,
        num_fused_shared_experts=1,
        shared_expert=None,
        shared_expert_gate=SimpleNamespace(weight=zeros(1, 256), bias=None),
        gate=SimpleNamespace(weight=zeros(16, 256)),
        topk=SimpleNamespace(
            topk_config=SimpleNamespace(
                top_k=10,
                use_grouped_topk=False,
                custom_routing_function=None,
                correction_bias=None,
                scoring_func="softmax",
                renormalize=True,
                num_fused_shared_experts=0,
                routed_scaling_factor=None,
            ),
            enable_waterfill=False,
        ),
        experts=Experts(),
        _append_shared_to_topk_output=append,
        appended=appended,
    )


@pytest.fixture
def ops(monkeypatch):
    """The three ops as fakes: each records (op, arguments) and returns what ``serves``
    names for it (a FallbackReason by default); the router returns its outputs."""
    state = SimpleNamespace(calls=[], serves=set())

    def fake(name):
        def op(**arguments):
            state.calls.append((name, arguments))
            if name not in state.serves:
                return FallbackReason.NO_WORKLOAD
            if name == "qwen2_moe_router":
                return arguments["topk_weights"], arguments["topk_ids"]
            return f"{name} out"

        monkeypatch.setattr(qwen2_moe, name, op)

    for name in OPS:
        fake(name)
    return state


def expert_arguments(layer, hidden):
    experts = layer.experts
    return {
        "hidden_states": hidden,
        "shared_gate_weight": layer.shared_expert_gate.weight,
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale_inv,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale_inv,
    }


def test_the_whole_block_reaches_one_op_at_its_point(sglang, ops):
    layer, hidden = block(), zeros(4, 256)
    ops.serves = {"qwen2_moe_fp8_router_experts"}
    out = qwen2_moe._forward_router_experts(layer, hidden)
    assert out == "qwen2_moe_fp8_router_experts out"
    ((name, arguments),) = ops.calls
    assert name == "qwen2_moe_fp8_router_experts"
    assert arguments == {
        **expert_arguments(layer, hidden),
        "router_weight": layer.gate.weight,
        "topk": 10,
    }
    point = contract_point(arguments, QWEN2_MOE_FP8_ROUTER_EXPERTS)
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    assert point.axes == {
        "n_tokens": 4,
        "model_dim": 256,
        "n_routed": 16,
        "inter_dim": 256,
    }


def test_the_router_and_experts_ops_serve_where_the_whole_block_does_not(sglang, ops):
    layer, hidden = block(), zeros(8, 256)
    ops.serves = {"qwen2_moe_router", "qwen2_moe_fp8_experts"}
    out = qwen2_moe._forward_router_experts(layer, hidden)
    assert out == "qwen2_moe_fp8_experts out"
    assert [name for name, _ in ops.calls] == list(OPS)
    (_, routing), (_, experts) = ops.calls[1:]
    weights, ids = routing["topk_weights"], routing["topk_ids"]
    assert (weights.shape, weights.dtype) == ((8, 10), torch.float32)
    assert (ids.shape, ids.dtype) == ((8, 10), torch.int32)
    assert routing["router_weight"] is layer.gate.weight
    assert experts == {
        **expert_arguments(layer, hidden),
        "topk_weights": weights,
        "topk_ids": ids,
    }
    assert contract_point(routing, QWEN2_MOE_ROUTER).axes["topk"] == 10
    assert contract_point(experts, QWEN2_MOE_FP8_EXPERTS).axes["n_experts"] == 17
    assert layer.experts.calls == [] and layer.appended == []


def test_sglang_runs_the_experts_on_the_router_ops_routing(sglang, ops):
    layer, hidden = block(), zeros(256, 256)
    ops.serves = {"qwen2_moe_router"}
    out = qwen2_moe._forward_router_experts(layer, hidden)
    assert out == "sglang experts"
    _, routing = ops.calls[1]
    ((topk_output, appended_hidden),) = layer.appended
    assert topk_output.topk_weights is routing["topk_weights"]
    assert topk_output.topk_ids is routing["topk_ids"]
    assert topk_output.router_logits is None and appended_hidden is hidden
    assert layer.experts.calls == [(hidden, "with the shared expert")]


def test_sglang_runs_the_block_where_the_router_op_does_not_serve(sglang, ops):
    layer, hidden = block(), zeros(3, 256)
    out = qwen2_moe._forward_router_experts(layer, hidden)
    assert out is FallbackReason.NO_WORKLOAD
    assert [name for name, _ in ops.calls] == list(OPS[:2])
    assert layer.experts.calls == [] and layer.appended == []


def _quant_method(layer):
    return layer.experts.quant_method


FALLBACKS = {
    "defer_finalize": lambda layer, state: "defer",
    "tuple input": lambda layer, state: "tuple",
    "shared expert apart": lambda layer, state: setattr(layer, "enable_shared_expert_fusion", False),
    "two shared slots": lambda layer, state: setattr(layer, "num_fused_shared_experts", 2),
    "shared expert MLP": lambda layer, state: setattr(layer, "shared_expert", object()),
    "shared gate bias": lambda layer, state: setattr(layer.shared_expert_gate, "bias", zeros(1)),
    "expert parallel": lambda layer, state: setattr(state.parallel, "moe_ep_size", 2),
    "grouped top-k": lambda layer, state: setattr(layer.topk.topk_config, "use_grouped_topk", True),
    "custom routing": lambda layer, state: setattr(layer.topk.topk_config, "custom_routing_function", len),
    "correction bias": lambda layer, state: setattr(layer.topk.topk_config, "correction_bias", zeros(16)),
    "sigmoid scores": lambda layer, state: setattr(layer.topk.topk_config, "scoring_func", "sigmoid"),
    "no renormalize": lambda layer, state: setattr(layer.topk.topk_config, "renormalize", False),
    "scaling factor": lambda layer, state: setattr(layer.topk.topk_config, "routed_scaling_factor", 2.5),
    "waterfill": lambda layer, state: setattr(layer.topk, "enable_waterfill", True),
    "deterministic": lambda layer, state: setattr(
        state.config.deterministic, "enable_deterministic_inference", True
    ),
    "eplb": lambda layer, state: setattr(state.config.moe, "enable_eplb", True),
    "redundant experts": lambda layer, state: setattr(state.config.moe, "ep_num_redundant_experts", 8),
    "expert location": lambda layer, state: setattr(state.config.moe, "init_expert_location", "random"),
    "distribution recorder": lambda layer, state: setattr(
        state.config.moe, "expert_distribution_recorder_mode", "stat"
    ),
    "routed experts capture": lambda layer, state: setattr(
        state.config.features, "enable_return_routed_experts", True
    ),
    "another quant method": lambda layer, state: setattr(
        layer.experts, "quant_method", SimpleNamespace(**vars(_quant_method(layer)))
    ),
    "triton runner": lambda layer, state: setattr(
        _quant_method(layer).runner.runner_backend, "is_aiter", lambda: False
    ),
    "mxfp8": lambda layer, state: setattr(_quant_method(layer), "use_mxfp8", True),
    "fp4 experts": lambda layer, state: setattr(_quant_method(layer), "is_fp4_expert", True),
    "per-channel scales": lambda layer, state: setattr(_quant_method(layer), "block_quant", False),
    "other blocks": lambda layer, state: setattr(_quant_method(layer), "weight_block_size", [128, 64]),
    "gelu": lambda layer, state: setattr(_quant_method(layer).runner.config, "activation", "gelu"),
    "swiglu limit": lambda layer, state: setattr(_quant_method(layer).runner.config, "swiglu_limit", 7.0),
    "expert biases": lambda layer, state: setattr(layer.experts, "w13_weight_bias", zeros(17, 512)),
    "expert mask": lambda layer, state: setattr(
        layer.experts.dispatcher, "expert_mask_gpu", zeros(17, dtype=torch.int32)
    ),
    "experts reduce": lambda layer, state: setattr(layer.experts, "reduce_results", True),
    "dwdp": lambda layer, state: setattr(layer.experts, "_dwdp_bound", True),
    "piecewise graph": lambda layer, state: setattr(state, "piecewise_cuda_graph", True),
    "breakable graph": lambda layer, state: setattr(state, "breakable_cuda_graph", True),
}


@pytest.mark.parametrize("case", FALLBACKS)
def test_the_block_falls_back_to_sglang_before_any_op(sglang, ops, case):
    layer, hidden, kwargs = block(), zeros(4, 256), {}
    change = FALLBACKS[case](layer, sglang)
    if change == "defer":
        kwargs["defer_finalize"] = True
    elif change == "tuple":
        hidden = (zeros(4, 256, dtype=FP8), zeros(4, 2, dtype=torch.float32))
    ops.serves = set(OPS)
    result = qwen2_moe._forward_router_experts(layer, hidden, **kwargs)
    assert result is FallbackReason.OUT_OF_CONTRACT
    assert ops.calls == [] and layer.experts.calls == []
