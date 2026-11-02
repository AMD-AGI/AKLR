# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M2's MoE ops: what the route tells them and when it falls back to sglang, what
the references compute, and the committed kernels."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.models.minimax_m2.api as minimax_m2
from aklr.core.contract import semantics
from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec
from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.plugins.sglang.models.minimax_m2.contract import (
    MINIMAX_M2_FP8_MOE,
    MINIMAX_M2_MXFP4_MOE,
)
from tests.op_case import OpCase

FP8 = torch.float8_e4m3fn
CONTRACTS = {"fp8": MINIMAX_M2_FP8_MOE, "mxfp4": MINIMAX_M2_MXFP4_MOE}
OPS = {"fp8": "minimax_m2_fp8_moe", "mxfp4": "minimax_m2_mxfp4_moe"}
IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/models/minimax_m2/impls"
# the routed experts, hidden and intermediate sizes of the shrunk MoE the route tests use
N_ROUTED, MODEL_DIM, INTER_DIM = 16, 256, 256


def contract_point(arguments, contract):
    """The Point of the call the route made, as dispatch keys it; None outside the contract."""
    codec = FastKeyCodec.from_contract(contract)
    call = {p.name: p.default for p in contract.api_params} | arguments
    key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()(call)
    return None if key is None else codec.to_point(key)


@pytest.fixture
def moe_sglang(sglang, monkeypatch):
    """The sglang stand-ins with the MoE flags the route reads: no all-to-all, no elastic
    expert-parallel join."""
    sglang.a2a_none = True
    sglang.config.moe.ep_join_mode = None
    module = ModuleType("sglang.srt.layers.moe")
    module.get_moe_a2a_backend = lambda: SimpleNamespace(is_none=lambda: sglang.a2a_none)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", module)
    return sglang


@pytest.fixture
def fake_ops(monkeypatch):
    """Every call the route makes to an op, as (op, arguments); each op returns 7s."""
    calls = []

    def fake(name):
        def op(**arguments):
            calls.append((name, arguments))
            return torch.full_like(arguments["hidden_states"], 7)

        monkeypatch.setattr(minimax_m2, name, op)

    for name in OPS.values():
        fake(name)
    return calls


def aiter_runner():
    """aiter's fused_moe runner as sglang configures it for MiniMaxM2MoE."""
    return SimpleNamespace(
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


def minimax_moe(precision="fp8", ep_size=1, ep_rank=0):
    """sglang's MiniMaxM2MoE shrunk: N_ROUTED experts, of which this rank holds
    N_ROUTED // ep_size from ep_rank's on, shuffled for aiter as sglang stores them; fp8 at
    128 x 128 blocks, or Quark's MXFP4 (uint8 storage); the fp32 gate and bias, sigmoid
    top-8 renormalized."""
    held = N_ROUTED // ep_size
    if precision == "fp8":
        w13 = torch.zeros(held, 2 * INTER_DIM, MODEL_DIM, dtype=FP8)
        w2 = torch.zeros(held, MODEL_DIM, INTER_DIM, dtype=FP8)
        method = type("Fp8MoEMethod", (SimpleNamespace,), {})(
            runner=aiter_runner(),
            block_quant=True,
            use_mxfp8=False,
            is_fp4_expert=False,
            weight_block_size=[128, 128],
        )
        quantization = {
            "quant_method": method,
            "w13_weight_scale_inv": torch.zeros(held, 2 * INTER_DIM // 128, 2),
            "w2_weight_scale_inv": torch.zeros(held, 2, INTER_DIM // 128),
        }
    else:
        w13 = torch.zeros(held, 2 * INTER_DIM, MODEL_DIM // 2, dtype=torch.uint8)
        w2 = torch.zeros(held, MODEL_DIM, INTER_DIM // 2, dtype=torch.uint8)
        scheme = type("QuarkW4A4MXFp4MoE", (SimpleNamespace,), {})(
            runner=aiter_runner(), static_input_scales=False
        )
        quantization = {
            "scheme": scheme,
            "quant_method": SimpleNamespace(),
            "w13_weight_scale": torch.zeros(held, 2 * INTER_DIM, MODEL_DIM // 32, dtype=torch.uint8),
            "w2_weight_scale": torch.zeros(held, MODEL_DIM, INTER_DIM // 32, dtype=torch.uint8),
        }
    w13.is_shuffled = w2.is_shuffled = True
    experts = type("FusedMoE", (SimpleNamespace,), {})(
        w13_weight=w13,
        w2_weight=w2,
        dispatcher=type("StandardDispatcher", (SimpleNamespace,), {})(expert_mask_gpu=None),
        reduce_results=False,
        num_fused_shared_experts=0,
        num_experts=N_ROUTED,
        moe_ep_size=ep_size,
        moe_ep_rank=ep_rank,
        _expert_storage_rank=ep_rank,
        _num_local_routed=held,
        use_presharded_weights=False,
        hidden_pad=0,
        intermediate_pad=0,
        **quantization,
    )
    bias = torch.zeros(N_ROUTED)
    routing = SimpleNamespace(
        top_k=8,
        use_grouped_topk=False,
        renormalize=True,
        scoring_func="sigmoid",
        correction_bias=bias,
        custom_routing_function=None,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    return SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(weight=torch.zeros(N_ROUTED, MODEL_DIM), bias=None),
        topk=SimpleNamespace(topk_config=routing, waterfill_balancer=None),
        e_score_correction_bias=bias,
    )


def hidden(n_tokens=4):
    return torch.zeros(n_tokens, MODEL_DIM, dtype=torch.bfloat16)


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
@pytest.mark.parametrize(("ep_size", "ep_rank"), [(1, 0), (4, 2)], ids=["all-experts", "ep4-rank2"])
def test_moe_reaches_its_op_with_sglangs_tensors(moe_sglang, fake_ops, precision, ep_size, ep_rank):
    moe, x = minimax_moe(precision, ep_size, ep_rank), hidden()
    out = minimax_m2._moe_forward_normal(moe, x)
    ((op, arguments),) = fake_ops
    assert op == OPS[precision]
    # forward_normal's result is this rank's share, summed over the ranks by sglang
    assert torch.equal(out, torch.full_like(x, 7))
    experts = moe.experts
    assert arguments["hidden_states"] is x
    assert arguments["router_weight"] is moe.gate.weight
    assert arguments["correction_bias"] is moe.e_score_correction_bias
    assert arguments["w1"] is experts.w13_weight
    assert arguments["w2"] is experts.w2_weight
    scales = (
        ("w13_weight_scale_inv", "w2_weight_scale_inv")
        if precision == "fp8"
        else ("w13_weight_scale", "w2_weight_scale")
    )
    assert arguments["w1_scale"] is getattr(experts, scales[0])
    assert arguments["w2_scale"] is getattr(experts, scales[1])
    assert arguments["topk"] == 8
    assert arguments["expert_offset"] == ep_rank * (N_ROUTED // ep_size)
    point = contract_point(arguments, CONTRACTS[precision])
    assert point is not None and point.layout_variant_name == "shuffled_16x16"
    assert point.axes["n_routed"] == N_ROUTED
    assert point.axes["n_experts"] == N_ROUTED // ep_size
    assert point.axes["inter_dim"] == INTER_DIM


def experts_of(moe):
    return moe.experts


def routing_of(moe):
    return moe.topk.topk_config


def runner_of(moe):
    experts = moe.experts
    return (getattr(experts, "scheme", None) or experts.quant_method).runner


# Each makes the shrunk MoE (of either precision) one the ops do not express.
FALLBACKS = {
    "gate-bias": lambda moe, sg: setattr(moe.gate, "bias", torch.zeros(N_ROUTED)),
    "no-correction-bias": lambda moe, sg: setattr(routing_of(moe), "correction_bias", None),
    "softmax": lambda moe, sg: setattr(routing_of(moe), "scoring_func", "softmax"),
    "not-renormalized": lambda moe, sg: setattr(routing_of(moe), "renormalize", False),
    "scaled": lambda moe, sg: setattr(routing_of(moe), "routed_scaling_factor", 2.5),
    "grouped": lambda moe, sg: setattr(routing_of(moe), "use_grouped_topk", True),
    "custom-routing": lambda moe, sg: setattr(routing_of(moe), "custom_routing_function", object()),
    "eplb": lambda moe, sg: setattr(sg.config.moe, "enable_eplb", True),
    "expert-recorder": lambda moe, sg: setattr(sg.config.moe, "expert_distribution_recorder_mode", "stat"),
    "routed-experts-capture": lambda moe, sg: setattr(
        sg.config.features, "enable_return_routed_experts", True
    ),
    "deterministic": lambda moe, sg: setattr(sg.config.deterministic, "enable_deterministic_inference", True),
    "all-to-all": lambda moe, sg: setattr(sg, "a2a_none", False),
    "dp-attention": lambda moe, sg: setattr(sg, "dp_attention", True),
    "not-fused-moe": lambda moe, sg: setattr(
        moe, "experts", type("DeepEPMoE", (SimpleNamespace,), {})(**vars(moe.experts))
    ),
    "other-dispatcher": lambda moe, sg: setattr(
        experts_of(moe), "dispatcher", SimpleNamespace(expert_mask_gpu=None)
    ),
    "reduced-inside": lambda moe, sg: setattr(experts_of(moe), "reduce_results", True),
    "shared-experts-fused": lambda moe, sg: setattr(experts_of(moe), "num_fused_shared_experts", 1),
    "redundant-experts": lambda moe, sg: setattr(experts_of(moe), "num_experts", N_ROUTED + 4),
    "elastic-join": lambda moe, sg: setattr(sg.config.moe, "ep_join_mode", "scale"),
    "triton-runner": lambda moe, sg: setattr(runner_of(moe).runner_backend, "is_aiter", lambda: False),
    "gelu": lambda moe, sg: setattr(runner_of(moe).config, "activation", "gelu"),
    "swiglu-limit": lambda moe, sg: setattr(runner_of(moe).config, "swiglu_limit", 7.0),
    "expert-bias": lambda moe, sg: setattr(experts_of(moe), "w13_weight_bias", torch.zeros(1)),
    "hidden-pad": lambda moe, sg: setattr(experts_of(moe), "hidden_pad", 128),
}
FP8_FALLBACKS = {
    "per-channel-scales": lambda moe: setattr(moe.experts.quant_method, "block_quant", False),
    "mxfp8": lambda moe: setattr(moe.experts.quant_method, "use_mxfp8", True),
    "fp4-experts": lambda moe: setattr(moe.experts.quant_method, "is_fp4_expert", True),
    "other-blocks": lambda moe: setattr(moe.experts.quant_method, "weight_block_size", [128, 64]),
    "gate-up-interleaved": lambda moe: setattr(moe.experts, "_aiter_gate_up_interleaved", True),
    "padded-intermediate": lambda moe: setattr(moe.experts, "intermediate_pad", 128),
}
MXFP4_FALLBACKS = {
    "static-input-scales": lambda moe: setattr(moe.experts.scheme, "static_input_scales", True),
    "presharded-padding": lambda moe: (
        setattr(moe.experts, "intermediate_pad", 128),
        setattr(moe.experts, "use_presharded_weights", True),
    ),
    "unquantized": lambda moe: (
        delattr(moe.experts, "scheme"),
        setattr(
            moe.experts,
            "quant_method",
            type("UnquantizedFusedMoEMethod", (SimpleNamespace,), {})(),
        ),
    ),
}


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
@pytest.mark.parametrize("change", FALLBACKS.values(), ids=list(FALLBACKS))
def test_moe_falls_back_to_sglang(moe_sglang, fake_ops, precision, change):
    moe = minimax_moe(precision, ep_size=4, ep_rank=1)
    change(moe, moe_sglang)
    result = minimax_m2._moe_forward_normal(moe, hidden())
    assert isinstance(result, FallbackReason) and fake_ops == []


@pytest.mark.parametrize(
    ("precision", "change"),
    [*(("fp8", c) for c in FP8_FALLBACKS.values())] + [*(("mxfp4", c) for c in MXFP4_FALLBACKS.values())],
    ids=[*FP8_FALLBACKS, *MXFP4_FALLBACKS],
)
def test_moe_falls_back_on_experts_off_the_ops(moe_sglang, fake_ops, precision, change):
    moe = minimax_moe(precision)
    change(moe)
    result = minimax_m2._moe_forward_normal(moe, hidden())
    assert isinstance(result, FallbackReason) and fake_ops == []


def test_a_padded_mxfp4_intermediate_stays_with_its_op(moe_sglang, fake_ops):
    """sglang's loader zeroes Quark's padding, so the op's sum over the stored width is
    sglang's own."""
    moe = minimax_moe("mxfp4")
    moe.experts.intermediate_pad = 128
    minimax_m2._moe_forward_normal(moe, hidden())
    assert [op for op, _ in fake_ops] == ["minimax_m2_mxfp4_moe"]


def test_an_empty_batch_falls_back(moe_sglang, fake_ops):
    result = minimax_m2._moe_forward_normal(minimax_moe(), hidden(0))
    assert isinstance(result, FallbackReason) and fake_ops == []


def test_the_ops_fallback_reaches_sglang(moe_sglang, monkeypatch):
    monkeypatch.setattr(minimax_m2, "minimax_m2_fp8_moe", lambda **_: FallbackReason.NO_WORKLOAD)
    result = minimax_m2._moe_forward_normal(minimax_moe(), hidden())
    assert result is FallbackReason.NO_WORKLOAD


def case_at(precision, n_routed, n_experts):
    """The small case of ``precision`` with n_routed router rows, n_experts held."""
    base = OpCase.of(CONTRACTS[precision].contract_id).point
    axes = {**base.axes, "n_routed": n_routed, "n_experts": n_experts}
    return OpCase(base.model_copy(update={"axes": axes}))


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_the_ranks_shares_sum_to_the_moe(precision):
    """Each of two expert-parallel ranks adds the routes to its half of the experts; their
    shares sum to the MoE on one rank."""
    whole = case_at(precision, 16, 16)
    args = whole.inputs()
    expected = whole.reference_outputs(args)["out"]
    half = case_at(precision, 16, 8)
    shares = []
    for rank in (0, 1):
        held = slice(8 * rank, 8 * rank + 8)
        rank_args = {
            **args,
            **{n: args[n][held] for n in ("w1", "w1_scale", "w2", "w2_scale")},
            "expert_offset": 8 * rank,
        }
        share = half.reference_outputs(rank_args)["out"]
        assert share.abs().sum() > 0
        shares.append(share)
    torch.testing.assert_close(shares[0] + shares[1], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_the_correction_bias_picks_experts_but_never_weighs_them(precision):
    """A bias the same on every expert picks the same experts at the same weights."""
    case = case_at(precision, 16, 16)
    args = case.inputs()
    shifted = {**args, "correction_bias": args["correction_bias"] + 4.0}
    torch.testing.assert_close(case.reference_outputs(shifted)["out"], case.reference_outputs(args)["out"])


def is_device_function(node):
    return any(ast.unparse(d) in ("gluon.jit", "triton.jit") for d in node.decorator_list)


def test_the_kernels_are_self_contained_openai_schemas():
    """23 Gluon kernels serve each of 40 exact MiniMax-M2.5 TP4 Points once: fp8 and MXFP4,
    every expert on the rank or 64 of 256 (EP4), all but MXFP4 EP1 at 32, 128 and 256 tokens
    (there the kernels only match sglang's aiter call, which skips the padded intermediate);
    every kernel passes the audit and reads no global next_power_of_2 in device code (sglang
    replaces it)."""
    registry = ContractRegistry.default()
    served = {}
    paths = sorted(IMPLS.rglob("*.json"))
    for path in paths:
        schema = registry.parse_schema(path.read_bytes(), str(path))
        kernel = schema.kernel
        assert kernel.language == "triton" and kernel.author == "openai"
        assert kernel.requires == {"triton": ">=3.8"}
        assert not kernel.state_bytes and not kernel.peer_bytes
        ((source,),) = [kernel.sources]
        assert source.path == "kernel.py"
        assert source.content.startswith("# Created by OpenAI\n")
        tree = ast.parse(source.content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and is_device_function(node):
                assert "triton.next_power_of_2" not in ast.unparse(node), node.name
        findings = check_kernel(
            str(path),
            kernel,
            call_keys=schema.kernel_call_keys,
            tensor_names=tuple(schema.tensors()),
            mutable_names=schema.mutable_tensors,
        )
        assert list(findings) == []
        point = schema.point(schema.workloads[0].axes)
        assert point.layout_variant_name == "shuffled_16x16"
        assert (point.axes["model_dim"], point.axes["n_routed"]) == (3072, 256)
        held, inter = point.axes["n_experts"], point.axes["inter_dim"]
        for workload in schema.workloads:
            key = (schema.contract_id, held, inter, workload.axes["n_tokens"])
            assert key not in served, key
            served[key] = path
            assert workload.scalars == {"expert_offset": 192 if held == 64 else 0}
    decode = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    expected = {
        ("minimax_m2_fp8_moe@1.0", 256, 384): decode,
        ("minimax_m2_fp8_moe@1.0", 64, 1536): decode,
        ("minimax_m2_mxfp4_moe@1.0", 256, 512): (1, 2, 4, 8, 16, 64, 1024, 6144, 8192),
        ("minimax_m2_mxfp4_moe@1.0", 64, 1536): (*decode, 2048, 4096, 6144, 8192),
    }
    assert set(served) == {(*k, m) for k, ms in expected.items() for m in ms}
    assert len(paths) == 23 and len(served) == 40


def test_the_kernels_read_mxfp4_as_bytes():
    """The MXFP4 kernels take fp4 and e8m0 tensors as sglang holds them, uint8, also when
    called with their typed views."""
    for path in sorted(IMPLS.rglob("*.json")):
        schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
        run = ast.parse(schema.kernel.sources[0].content).body[-1]
        assert isinstance(run, ast.FunctionDef) and run.name == "run"
        views = ast.unparse(run).count(".view(torch.uint8)")
        assert views == (4 if schema.contract_id == "minimax_m2_mxfp4_moe@1.0" else 0)


def test_the_small_points_draw_finite_inputs():
    """The small cases' inputs decode to finite values (bench draws them so)."""
    for precision in ("fp8", "mxfp4"):
        case = case_at(precision, 16, 8)
        point: PointSpec = case.point
        for name, value in case.inputs().items():
            if torch.is_tensor(value):
                assert semantics.to_float64(value, point.dtypes[name]).isfinite().all()
