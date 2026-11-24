# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR serving aiter's fused MoE (``fused_moe``) and its grouped top-k (``biased_grouped_topk``) as aiter's
kernel plugin: what each selector takes, what it declines, the AKLR call its run makes, the sort a caller
registered for aiter's fused MoE, and, with aiter's plugin interface, the route aiter takes."""

import enum
import importlib
import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.situv2_a4w4_moe.api import situv2_a4w4_moe

provider = importlib.import_module("aklr.plugins.aiter.provider")
base = importlib.import_module("aklr.plugins.aiter.provider._base")
moe = importlib.import_module("aklr.plugins.aiter.provider.moe")
topk = importlib.import_module("aklr.plugins.aiter.provider.topk")
aklr_fused_moe = importlib.import_module("aklr.ops.fused_moe.api").fused_moe

BF16, F32, FP8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
FP4 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
I32 = torch.int32
KERNEL = RunnableMetadata(
    build_type="flydsl", contract_id="any@1.0", schema_name="any_schema", content_hash="0123456789abcdef0123"
)
HERD = "AITER_FLYDSL_USE_HERD"

# aiter's contracts (ROCm/aiter tests/plugin_contracts.json, branch kyle/aklr-routes-moe): version, positional
# parameters
AITER_CONTRACTS = {
    "fused_moe": (
        1,
        (
            "hidden_states",
            "w1",
            "w2",
            "topk_weight",
            "topk_ids",
            "expert_mask",
            "activation",
            "quant_type",
            "doweight_stage1",
            "w1_scale",
            "w2_scale",
            "a1_scale",
            "a2_scale",
            "block_size_M",
            "num_local_tokens",
            "moe_sorting_dispatch_policy",
            "dtype",
            "hidden_pad",
            "intermediate_pad",
            "bias1",
            "bias2",
            "splitk",
            "swiglu_limit",
            "beta",
            "linear_beta",
            "gate_mode",
            "ep_arena_handle",
            "ep_combine_input_offset",
            "ep_slot_stride_bytes",
            "ep_max_tokens_per_rank",
            "ep_world_size",
            "ep_combine_quant",
            "ep_source_token_map",
            "output",
            "quant_type_a",
            "quant_dtype_a",
            "quant_dtype_a2",
            "herd_topk",
            "herd_routed_scaling_factor",
            "deterministic",
        ),
    ),
    "biased_grouped_topk": (
        1,
        (
            "gating_output",
            "correction_bias",
            "topk_weights",
            "topk_ids",
            "num_expert_group",
            "topk_group",
            "need_renorm",
            "routed_scaling_factor",
            "num_valid_rows",
        ),
    ),
}
EP_FIELDS = (
    "ep_arena_handle",
    "ep_combine_input_offset",
    "ep_slot_stride_bytes",
    "ep_max_tokens_per_rank",
    "ep_world_size",
    "ep_combine_quant",
)


class ActivationType(enum.IntEnum):
    """aiter's enums by name; the values are the stand-in's own."""

    Silu = 0
    Gelu = 1
    Swiglu = 2
    Situv2 = 3


class QuantType(enum.IntEnum):
    No = 0
    per_1x32 = 1
    per_1x128 = 2


# aiter's defaults, by parameter name: aiter applies them before asking
DEFAULTS = {
    "fused_moe": {
        "expert_mask": None,
        "activation": ActivationType.Silu.value,
        "quant_type": QuantType.No.value,
        "doweight_stage1": False,
        "w1_scale": None,
        "w2_scale": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_size_M": -1,
        "num_local_tokens": None,
        "moe_sorting_dispatch_policy": 0,
        "dtype": None,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "bias1": None,
        "bias2": None,
        "splitk": 0,
        "swiglu_limit": None,
        "beta": None,
        "linear_beta": None,
        "gate_mode": "separated",
        **dict.fromkeys(EP_FIELDS, 0),
        "ep_source_token_map": None,
        "output": None,
        "quant_type_a": None,
        "quant_dtype_a": None,
        "quant_dtype_a2": None,
        "herd_topk": 0,
        "herd_routed_scaling_factor": 1.0,
        "deterministic": False,
    },
    "biased_grouped_topk": {"routed_scaling_factor": 1.0, "num_valid_rows": None},
}


def _served_calls():
    """A call AKLR's contract takes, per aiter op, by parameter name, defaults applied."""
    z = torch.zeros
    return {
        "fused_moe": {
            **DEFAULTS["fused_moe"],
            "hidden_states": z(4, 256, dtype=BF16),
            "w1": z(2, 512, 128, dtype=torch.uint8).view(FP4),
            "w2": z(2, 256, 128, dtype=torch.uint8).view(FP4),
            "topk_weight": z(4, 2),
            "topk_ids": z(4, 2, dtype=I32),
            "quant_type": QuantType.per_1x32.value,
            "w1_scale": z(2, 512, 8, dtype=torch.uint8),
            "w2_scale": z(2, 256, 4, dtype=torch.uint8),
        },
        "biased_grouped_topk": {
            **DEFAULTS["biased_grouped_topk"],
            "gating_output": z(4, 8, dtype=BF16),
            "correction_bias": z(8, dtype=BF16),
            "topk_weights": z(4, 2),
            "topk_ids": z(4, 2, dtype=I32),
            "num_expert_group": 1,
            "topk_group": 1,
            "need_renorm": True,
        },
    }


def _args(op: str, **changes):
    call = _served_calls()[op]
    assert set(changes) <= set(call), f"not a parameter of aiter's {op}: {set(changes) - set(call)}"
    call |= changes
    return tuple(call[name] for name in AITER_CONTRACTS[op][1])


def _select(op: str, **changes):
    return base.SELECTORS[op][1](_args(op, **changes), {})


def _api(fn):
    """A stand-in AKLR api calling ``fn``, whose ``serving`` answers KERNEL."""
    fn.serving = lambda *a, **k: KERNEL
    return fn


def _same(left, right) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


@pytest.fixture
def stand_in_aiter(monkeypatch):
    """A stand-in aiter package: its enums, the activation dtype its fused MoE picks (``picked``, fp8 to
    start) and the sort a caller registers for its next fused MoE."""
    package = ModuleType("aiter")
    package.ActivationType, package.QuantType = ActivationType, QuantType
    package.dtypes = SimpleNamespace(fp8=FP8, fp4x2=FP4, bf16=BF16)
    state = SimpleNamespace(package=package, picked=FP8)

    fused_moe = ModuleType("aiter.fused_moe")
    fused_moe.resolve_activation_dtype = lambda quant_type, q_dtype_w, **kwargs: state.picked

    presorted_moe = ModuleType("aiter.presorted_moe")
    local = threading.local()

    def pop_presorted_moe():
        aux = getattr(local, "aux", None)
        local.aux = None
        return aux

    presorted_moe.register_presorted_moe = lambda aux: setattr(local, "aux", aux)
    presorted_moe.pop_presorted_moe = pop_presorted_moe
    state.register = presorted_moe.register_presorted_moe
    state.registered = lambda: getattr(local, "aux", None)

    monkeypatch.setitem(sys.modules, "aiter", package)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.presorted_moe", presorted_moe)
    monkeypatch.delenv(HERD, raising=False)
    moe._takes_gate_up_interleave.cache_clear()
    yield state
    moe._takes_gate_up_interleave.cache_clear()


@pytest.fixture
def aklr_serves(monkeypatch, stand_in_aiter):
    """Every AKLR api the selectors ask answers with KERNEL; a candidate is (kernel, run)."""
    for module in (moe, topk):
        monkeypatch.setattr(
            module,
            "offer",
            lambda serving, run: None if isinstance(serving, FallbackReason) else (serving, run),
        )
        for api_fn in vars(module).values():
            if callable(getattr(api_fn, "serving", None)):
                monkeypatch.setattr(api_fn, "serving", lambda *a, **k: KERNEL)
    return stand_in_aiter


def test_each_selector_serves_its_aiter_contract_at_its_version():
    for op, (version, _) in AITER_CONTRACTS.items():
        assert base.SELECTORS[op][0] == version
        assert provider.provider.op_versions[op] == {version}


SERVED = [
    ("fused_moe", {}),
    ("fused_moe", {"output": torch.empty(4, 256, dtype=BF16), "dtype": BF16}),
    ("fused_moe", {"expert_mask": torch.ones(4, dtype=I32), "num_local_tokens": torch.ones(1, dtype=I32)}),
    # schedules aiter's sort
    ("fused_moe", {"moe_sorting_dispatch_policy": 1}),
    # scales herd_topk's candidates alone
    ("fused_moe", {"herd_routed_scaling_factor": 2.5}),
    ("fused_moe", {"activation": ActivationType.Situv2.value}),
    ("biased_grouped_topk", {}),
    ("biased_grouped_topk", {"num_expert_group": 4, "topk_group": 4, "routed_scaling_factor": 2.5}),
]


@pytest.mark.parametrize(("op", "changes"), SERVED)
def test_a_served_call_offers_a_run_taking_aiters_parameters_by_name(aklr_serves, op, changes):
    """aiter hands run every positional parameter: run names them as aiter does, in aiter's order."""
    kernel, run = _select(op, **changes)
    assert kernel is KERNEL
    parameters = inspect.signature(run).parameters
    assert tuple(parameters) == AITER_CONTRACTS[op][1]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty for p in parameters.values())
    inspect.signature(run).bind(*_args(op, **changes))


DECLINED = [
    ("fused_moe", {"block_size_M": 32}),
    ("fused_moe", {"hidden_pad": 128}),
    ("fused_moe", {"intermediate_pad": 128}),
    ("fused_moe", {"a1_scale": torch.ones(1)}),
    ("fused_moe", {"a2_scale": torch.ones(1)}),
    ("fused_moe", {"bias1": torch.ones(1)}),
    ("fused_moe", {"bias2": torch.ones(1)}),
    # the stage-2 scatter writes the rows into the peers' buffers
    ("fused_moe", {"ep_source_token_map": torch.zeros(4, dtype=I32)}),
    *[("fused_moe", {field: 1}) for field in EP_FIELDS],
    ("fused_moe", {"quant_type_a": QuantType.per_1x32.value}),
    ("fused_moe", {"quant_dtype_a": FP8}),
    ("fused_moe", {"quant_dtype_a2": FP8}),
    ("fused_moe", {"herd_topk": 8}),
    # AKLR's order of the top-k sums is not fixed
    ("fused_moe", {"deterministic": True}),
    ("biased_grouped_topk", {"num_valid_rows": torch.ones(1, dtype=I32)}),
    # aiter drops groups, AKLR keeps every one
    ("biased_grouped_topk", {"num_expert_group": 4, "topk_group": 2}),
    ("biased_grouped_topk", {"correction_bias": torch.zeros(8, dtype=F32)}),
]


@pytest.mark.parametrize(
    ("op", "changes"), DECLINED, ids=lambda v: v if isinstance(v, str) else next(iter(v))
)
def test_a_call_outside_aklrs_contract_is_declined_before_aklr_is_asked(aklr_serves, op, changes):
    assert _select(op, **changes) is None


def test_situv2_is_declined_where_aiter_keeps_its_activations_in_bf16(aklr_serves):
    aklr_serves.picked = BF16
    assert _select("fused_moe", activation=ActivationType.Situv2.value) is None


SITUV2_A4W4 = {"activation": ActivationType.Situv2.value, "beta": 1.5, "linear_beta": 2.5}


def test_situv2_on_mxfp4_activations_is_served_by_situv2_a4w4_moe(monkeypatch, aklr_serves):
    """vLLM's Kimi-K3 (AITER_SITUV2_A4W4): aiter rounds SiTUv2's activations to MXFP4."""
    aklr_serves.picked = FP4
    calls, asked = [], []
    api = _api(lambda *a, **k: calls.append((a, k)) or "y")
    api.serving = lambda *a, **k: asked.append(k) or KERNEL
    monkeypatch.setattr(moe, "situv2_a4w4_moe", api)
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: pytest.fail("fused_moe asked")))
    out = torch.empty(4, 256, dtype=BF16)
    changes = {**SITUV2_A4W4, "output": out, "dtype": BF16}
    kernel, run = _select("fused_moe", **changes)
    assert kernel is KERNEL
    args = _args("fused_moe", **changes)
    assert run(*args) == "y"
    ((tensors, kwargs),) = calls
    # asked with what it runs on
    ((selected),) = asked
    assert _same(tensors, args[:5]) and selected.keys() == kwargs.keys()
    assert kwargs["w1_scale"] is args[9] and kwargs["w2_scale"] is args[10]
    assert (kwargs["beta"], kwargs["linear_beta"], kwargs["out_dtype"]) == (1.5, 2.5, BF16)
    assert kwargs["out"] is out
    inspect.signature(situv2_a4w4_moe).bind(*tensors, **kwargs)


class _PybindEnum:
    """aiter's enums are pybind11's: each lookup a new object, equal by value, never the same object."""

    def __init__(self, value):
        self.value = int(value)

    def __eq__(self, other):
        return getattr(other, "value", other) == self.value

    def __hash__(self):
        return hash(self.value)


def test_situv2_on_mxfp4_activations_is_told_by_value_from_aiters_pybind_enums(monkeypatch, aklr_serves):
    activation, quant = type("ActivationType", (_PybindEnum,), {}), type("QuantType", (_PybindEnum,), {})
    activation.Situv2, quant.per_1x32 = activation(ActivationType.Situv2), quant(QuantType.per_1x32)
    monkeypatch.setattr(aklr_serves.package, "ActivationType", activation)
    monkeypatch.setattr(aklr_serves.package, "QuantType", quant)
    aklr_serves.picked = FP4
    monkeypatch.setattr(moe, "situv2_a4w4_moe", _api(lambda *a, **k: "y"))
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: pytest.fail("fused_moe asked")))
    kernel, run = _select("fused_moe", **SITUV2_A4W4)
    assert kernel is KERNEL and run(*_args("fused_moe", **SITUV2_A4W4)) == "y"


@pytest.mark.parametrize(
    "changes",
    [
        {"expert_mask": torch.ones(2, dtype=I32)},
        {"num_local_tokens": torch.ones(1, dtype=I32)},
        {"doweight_stage1": True},
        {"swiglu_limit": 7.0},
        {"beta": None},
        {"gate_mode": "interleaved"},
    ],
    ids=["expert_mask", "num_local_tokens", "doweight_stage1", "swiglu_limit", "no_beta", "interleaved"],
)
def test_situv2_on_mxfp4_activations_outside_its_contract_is_declined(aklr_serves, changes):
    aklr_serves.picked = FP4
    assert _select("fused_moe", **{**SITUV2_A4W4, **changes}) is None


def test_a_one_group_call_aiter_routes_by_herd_is_declined(monkeypatch, aklr_serves):
    """``AITER_FLYDSL_USE_HERD=1`` has aiter drop routes by batch popularity for one group: not AKLR's top-k."""
    monkeypatch.setenv(HERD, "1")
    assert _select("biased_grouped_topk") is None
    assert _select("biased_grouped_topk", num_expert_group=4, topk_group=4) is not None


@pytest.mark.parametrize("op", list(AITER_CONTRACTS))
def test_a_call_aklr_falls_back_on_is_declined(monkeypatch, aklr_serves, op):
    for module in (moe, topk):
        for api_fn in vars(module).values():
            if callable(getattr(api_fn, "serving", None)):
                monkeypatch.setattr(api_fn, "serving", lambda *a, **k: FallbackReason.NO_WORKLOAD)
    assert _select(op) is None


def test_fused_moes_run_hands_aklr_aiters_enums_buffer_and_dtype(monkeypatch, aklr_serves):
    """aiter's ints go back to its enums, which AKLR reads by name; the result is AKLR's, the caller's
    buffer when it passed one; vLLM's 0.0 is no limit."""
    calls = []
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: calls.append((a, k)) or "y"))
    out = torch.empty(4, 256, dtype=BF16)
    changes = {"output": out, "dtype": BF16, "swiglu_limit": 0.0, "expert_mask": torch.ones(4, dtype=I32)}
    _, run = _select("fused_moe", **changes)
    args = _args("fused_moe", **changes)
    assert run(*args) == "y"
    ((tensors, kwargs),) = calls
    assert _same(tensors, args[:5])
    assert kwargs["activation"] is ActivationType.Silu and kwargs["quant_type"] is QuantType.per_1x32
    assert kwargs["out"] is out and kwargs["out_dtype"] is BF16 and kwargs["swiglu_limit"] is None
    assert kwargs["expert_mask"] is changes["expert_mask"] and kwargs["act_quant"] == 4
    inspect.signature(aklr_fused_moe).bind(*tensors, **kwargs)


def test_fused_moes_run_takes_situv2s_fp8_activations(monkeypatch, aklr_serves):
    calls = []
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: calls.append(k) or "y"))
    changes = {"activation": ActivationType.Situv2.value, "beta": 1.5, "linear_beta": 2.5}
    _, run = _select("fused_moe", **changes)
    run(*_args("fused_moe", **changes))
    (kwargs,) = calls
    assert kwargs["activation"] is ActivationType.Situv2 and kwargs["act_quant"] == 8
    assert (kwargs["beta"], kwargs["linear_beta"]) == (1.5, 2.5)


def test_biased_grouped_topks_run_writes_in_place_and_returns_none(monkeypatch, aklr_serves):
    """aiter's op returns None, which its registration as a torch op holds it to; the results are the two
    buffers AKLR writes."""
    calls = []
    monkeypatch.setattr(topk, "biased_grouped_topk", _api(lambda *a: calls.append(a) or (a[2], a[3])))
    changes = {"num_expert_group": 4, "topk_group": 4, "need_renorm": False, "routed_scaling_factor": 2.5}
    _, run = _select("biased_grouped_topk", **changes)
    args = _args("biased_grouped_topk", **changes)
    assert run(*args) is None
    (call,) = calls
    assert _same(call[:4], args[:4]) and call[4:] == (4, 4, False, 2.5)


def test_selecting_leaves_a_registered_sort_for_the_call(aklr_serves):
    aklr_serves.register({"M": 4})
    assert _select("fused_moe") is not None
    assert aklr_serves.registered() == {"M": 4}


def test_aklrs_run_drops_the_registered_sort_before_aklr_runs(monkeypatch, aklr_serves):
    """aiter consumes it beneath its route; left registered it would apply to aiter's next call."""
    seen = []
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: seen.append(aklr_serves.registered()) or "y"))
    aklr_serves.register({"M": 4})
    _, run = _select("fused_moe")
    run(*_args("fused_moe"))
    assert seen == [None] and aklr_serves.registered() is None


def test_a_failing_run_drops_the_registered_sort_too(monkeypatch, aklr_serves):
    def fails(*a, **k):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(moe, "fused_moe", _api(fails))
    aklr_serves.register({"M": 4})
    _, run = _select("fused_moe")
    with pytest.raises(RuntimeError, match="kernel failed"):
        run(*_args("fused_moe"))
    assert aklr_serves.registered() is None


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        return self.target


def _native(op, ran):
    """A function of aiter's signature for ``op``, with its defaults, recording that aiter computed the call."""
    names = AITER_CONTRACTS[op][1]
    defaults = DEFAULTS[op]
    parameters = ", ".join(f"{n}=_d[{n!r}]" if n in defaults else n for n in names)
    scope = {"_d": defaults, "_ran": ran}
    exec(f"def {op}({parameters}):\n    return _ran({op!r})\n", scope)
    return scope[op]


@pytest.fixture
def aiter_plugin():
    """aiter's plugin interface, imported before the stand-in takes the name of its package."""
    return pytest.importorskip("aiter.plugin")


@pytest.fixture
def aiter_routes(monkeypatch, aiter_plugin, stand_in_aiter):
    """aiter's plugin interface with AKLR enabled. An op aiter routes itself keeps aiter's route and contract,
    asked with explain() and never run (aiter's would build its kernels); each other is routed to a native of
    this file's contract, which, for the fused MoE, consumes the registered sort as aiter's does. Yields
    aiter's own routed ops, the test's routes and the ops their natives computed."""
    plugin = aiter_plugin
    stand_in_aiter.package.plugin = plugin
    plugin._reset_for_testing()
    installed = {op: plugin._routed_ops[op] for op in AITER_CONTRACTS if op in plugin._routed_ops}
    ran = []

    def native(op):
        ran.append(op)
        if op == "fused_moe":
            sys.modules["aiter.presorted_moe"].pop_presorted_moe()
            return "aiter"

    routed = {
        op: plugin.route(op=op, version=version)(_native(op, native))
        for op, (version, _) in AITER_CONTRACTS.items()
        if op not in installed
    }
    monkeypatch.setattr(plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider.provider),))
    monkeypatch.setattr(provider, "ready", lambda: True)
    provider.enable("test")
    yield SimpleNamespace(plugin=plugin, installed=installed, routed=routed, ran=ran)
    plugin._reset_for_testing()


def test_this_files_contracts_are_those_of_the_aiter_routing_them(aiter_routes):
    """The positional parameters and version of each op aiter routes itself are this file's, and so the
    provider's: aiter hands run the call by them."""
    for op, routed_op in aiter_routes.installed.items():
        version, names = AITER_CONTRACTS[op]
        positional = tuple(
            p.name for p in routed_op.contract.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD
        )
        assert (routed_op.op.version, positional) == (version, names), op


def test_aiter_routes_the_calls_aklr_serves_to_aklr(monkeypatch, aiter_routes, stand_in_aiter):
    """aiter checks that run takes the contract before handing it a call; the public fused_moe calls the
    routed layer by keyword, as here."""
    calls = []
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: calls.append("fused_moe") or "y"))
    monkeypatch.setattr(topk, "biased_grouped_topk", _api(lambda *a: calls.append("biased_grouped_topk")))
    plugin, routed = aiter_routes.plugin, aiter_routes.routed
    stand_in_aiter.register({"M": 4})

    for op in AITER_CONTRACTS:
        decision = plugin.explain(op, *_args(op)).decision
        assert (decision.route, decision.provider, decision.kernel_id) == (
            "provider",
            "aklr",
            KERNEL.kernel_ref,
        )
    assert stand_in_aiter.registered() == {"M": 4}  # explain runs nothing

    if "fused_moe" in routed:
        call = dict(zip(AITER_CONTRACTS["fused_moe"][1], _args("fused_moe")))
        assert routed["fused_moe"](**call) == "y"
        assert stand_in_aiter.registered() is None
    if "biased_grouped_topk" in routed:
        assert routed["biased_grouped_topk"](*_args("biased_grouped_topk")[:7]) is None
    assert calls == [op for op in AITER_CONTRACTS if op in routed] and aiter_routes.ran == []


def test_aiter_computes_the_calls_aklr_declines(monkeypatch, aiter_routes, stand_in_aiter):
    monkeypatch.setattr(moe, "fused_moe", _api(lambda *a, **k: pytest.fail("AKLR ran a declined call")))
    plugin, routed = aiter_routes.plugin, aiter_routes.routed
    declined = {
        "fused_moe": _args("fused_moe", deterministic=True),
        "biased_grouped_topk": _args("biased_grouped_topk", num_valid_rows=torch.ones(1)),
    }
    for op, args in declined.items():
        assert plugin.explain(op, *args).decision.route == "native"
    stand_in_aiter.register({"M": 4})
    if "fused_moe" in routed:
        assert routed["fused_moe"](*declined["fused_moe"]) == "aiter"
        assert stand_in_aiter.registered() is None
    if "biased_grouped_topk" in routed:
        routed["biased_grouped_topk"](*declined["biased_grouped_topk"])
    assert aiter_routes.ran == [op for op in declined if op in routed]
