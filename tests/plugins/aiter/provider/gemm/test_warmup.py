# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the framework plugins' GEMM warm-ups share (CPU, a stand-in aiter): the scratch call of aiter's GEMMs at a
Point and who answered it, the dispatchers that serve something, the Points of their table, and a model's weight
shapes."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.runtime.dispatch import DispatcherRegistry
from aklr.plugins.aiter.provider.gemm import warmup
from tests.op_case import OpCase

CPU = torch.device("cpu")
FP8 = torch.float8_e4m3fn


def point(m=3, n=4, k=8, has_bias=False, layout="b_shuffled", **dtypes):
    names = {"a": "bfloat16", "b": "bfloat16", "bias": "bfloat16", "out": "bfloat16"} | dtypes
    return SimpleNamespace(
        axes={"m": m, "n": n, "k": k}, attrs={"has_bias": has_bias}, dtypes=names, layout_variant_name=layout
    )


@pytest.fixture
def aiter(monkeypatch):
    """A stand-in aiter: its two GEMMs keep their arguments; ``decision`` is what its plugin interface answers."""
    state = SimpleNamespace(calls=[], scaled_calls=[], decision=SimpleNamespace(provider="aklr"))
    package = ModuleType("aiter")
    package.__path__ = []
    package.plugin = SimpleNamespace(last_decision=lambda: state.decision)
    tuned_gemm, ops, a8w8 = (
        ModuleType("aiter.tuned_gemm"),
        ModuleType("aiter.ops"),
        ModuleType("aiter.ops.gemm_op_a8w8"),
    )
    ops.__path__ = []
    tuned_gemm.gemm_a16w16 = lambda *args: state.calls.append(args)
    a8w8.gemm_a8w8_bpreshuffle = lambda *args, **kwargs: state.scaled_calls.append((args, kwargs))
    for name, module in (
        ("aiter", package),
        ("aiter.tuned_gemm", tuned_gemm),
        ("aiter.ops", ops),
        ("aiter.ops.gemm_op_a8w8", a8w8),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    state.package = package
    return state


@pytest.mark.parametrize("has_bias", [False, True])
def test_the_scratch_call_is_aiters_gemm_on_zeros_of_the_points_shapes(aiter, has_bias):
    """a [m, k], b [n, k] (aiter takes the weight no other way), a bias [n] where the Point has one, the output
    dtype; AKLR answered it."""
    assert warmup.call_once(point(has_bias=has_bias, out="float16"), CPU) is True
    ((a, b, bias, out_dtype),) = aiter.calls
    assert (a.shape, b.shape, a.dtype, b.dtype) == ((3, 8), (4, 8), torch.bfloat16, torch.bfloat16)
    assert not a.any() and not b.any() and out_dtype is torch.float16
    assert bias is None if not has_bias else (bias.shape, bias.dtype) == ((4,), torch.bfloat16)


@pytest.mark.parametrize(
    "decision",
    [None, SimpleNamespace(provider="another"), SimpleNamespace(provider=None)],
    ids=["none", "another", "aiter"],
)
def test_a_call_aiter_or_another_provider_answered_is_not_aklrs(aiter, decision):
    aiter.decision = decision
    assert warmup.call_once(point(), CPU) is False
    assert len(aiter.calls) == 1


def test_an_aiter_without_the_plugin_interface_answers_itself(aiter):
    del aiter.package.plugin
    assert warmup.call_once(point(), CPU) is False
    assert (
        warmup.call_once_a8w8(
            point(a="float8_e4m3fn", b="float8_e4m3fn", a_scale="float32", b_scale="float32"), CPU
        )
        is False
    )
    assert len(aiter.calls) == 1 and len(aiter.scaled_calls) == 1


def test_the_per_token_fp8_call_is_aiters_shuffled_gemm_with_unit_scales(aiter):
    scaled = point(a="float8_e4m3fn", b="float8_e4m3fn", a_scale="float32", b_scale="float32")
    assert warmup.call_once_a8w8(scaled, CPU) is True
    (((a, b, a_scale, b_scale), kwargs),) = aiter.scaled_calls
    assert (a.shape, b.shape, a.dtype, b.dtype) == ((3, 8), (4, 8), FP8, FP8)
    assert (a_scale.shape, b_scale.shape) == ((3, 1), (4, 1)) and a_scale.dtype is torch.float32
    assert bool((a_scale == 1).all()) and bool((b_scale == 1).all()) and kwargs == {"dtype": torch.bfloat16}


@pytest.mark.parametrize(
    "changed", [{"layout": "default"}, {"has_bias": True}], ids=["unshuffled", "with a bias"]
)
def test_a_per_token_fp8_point_aiter_has_no_op_for_is_not_called(aiter, changed):
    """aiter's op takes a shuffled weight without a bias alone."""
    assert warmup.call_once_a8w8(point(**changed), CPU) is False
    assert aiter.scaled_calls == [] and aiter.calls == []


@pytest.mark.parametrize(
    ("get", "contract_id"),
    [(warmup.gemm_dispatcher, "gemm@1.0"), (warmup.gemm_a8w8_dispatcher, "gemm_a8w8@1.0")],
    ids=["gemm", "gemm_a8w8"],
)
def test_a_dispatcher_is_given_only_where_it_serves_something(monkeypatch, get, contract_id):
    dispatchers = {}
    monkeypatch.setattr(
        DispatcherRegistry, "default", lambda: SimpleNamespace(dispatchers=lambda: dispatchers)
    )
    assert get() is None  # this device holds none
    dispatchers[contract_id] = SimpleNamespace(table=None)
    assert get() is None  # nothing committed for the device
    dispatchers[contract_id] = serving = SimpleNamespace(table=SimpleNamespace(runnables={}))
    assert get() is serving


def test_the_served_points_are_the_tables_keys_decoded():
    case = OpCase.of("gemm@1.0")
    key = FastKeyCodec.from_contract(case.contract).to_fast_key(case.point)
    dispatcher = SimpleNamespace(contract=case.contract, table=SimpleNamespace(runnables={key: object()}))
    (served,) = warmup.served_gemm_points(dispatcher)
    assert served.axes == case.point.axes and served.attrs == case.point.attrs
    assert served.dtypes == case.point.dtypes


def test_the_weight_shapes_are_every_matrix_either_way_round():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 8))
            self.bias = torch.nn.Parameter(torch.zeros(4))
            self.register_buffer("table", torch.zeros(2, 6))

    assert warmup.weight_shapes([Model(), None, "not a module"]) == {(4, 8), (8, 4), (2, 6), (6, 2)}
