# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR's GEMM warm-up after sglang's graph capture: which served kernels it calls, up to which size, and that
export records none of the calls."""

from types import SimpleNamespace

import pytest
import torch

from aklr.plugins.sglang import warmup


def point(m, n, k, trans_b=True):
    return SimpleNamespace(axes={"m": m, "n": n, "k": k}, attrs={"trans_b": trans_b})


class Model(torch.nn.Module):
    def __init__(self, *shapes):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(s)) for s in shapes])


def runner(model, **server_args):
    return SimpleNamespace(model=model, server_args=SimpleNamespace(**server_args))


@pytest.fixture
def served(monkeypatch):
    """A dispatcher serving the test's Points; the Points called, each with the recorder it saw, and the dispatchers
    whose Points were asked for."""
    state = SimpleNamespace(
        points=[],
        scaled_points=[],
        calls=[],
        scaled_calls=[],
        asked=[],
        dispatcher=SimpleNamespace(recorder="recorder"),
        scaled=None,  # a test with per-token fp8 Points sets a dispatcher of its own
    )
    monkeypatch.setattr(warmup, "gemm_dispatcher", lambda: state.dispatcher)
    monkeypatch.setattr(warmup, "gemm_a8w8_dispatcher", lambda: state.scaled)

    def points(dispatcher):
        state.asked.append(dispatcher)
        return state.points if dispatcher is state.dispatcher else state.scaled_points

    def call_once(p, device):
        state.calls.append((p.axes["m"], p.axes["n"], p.axes["k"], state.dispatcher.recorder))
        return True

    def call_once_a8w8(p, device):
        state.scaled_calls.append((p.axes["m"], p.axes["n"], p.axes["k"], state.scaled.recorder))
        return True

    monkeypatch.setattr(warmup, "served_gemm_points", points)
    monkeypatch.setattr(warmup, "call_once", call_once)
    monkeypatch.setattr(warmup, "call_once_a8w8", call_once_a8w8)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return state


def test_the_runner_models_weight_shapes_up_to_the_prefill_budget_are_warmed(served):
    served.points = [
        point(16, 32, 48),  # the model's weight
        point(16, 48, 32),  # the same weight, the other way round
        point(64, 32, 48),  # at the budget
        point(16, 100, 200),  # no weight of this runner's
        point(65, 32, 48),  # past the budget: no prefill step is that large
    ]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=64, chunked_prefill_size=32))
    assert [c[:3] for c in served.calls] == [(16, 32, 48), (16, 48, 32), (64, 32, 48)]


def test_a_kernel_aiter_cannot_ask_for_is_not_warmed(served):
    """aiter's GEMMs take the weight as [n, k] alone."""
    served.points = [point(16, 32, 48, trans_b=False)]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=64, chunked_prefill_size=32))
    assert served.calls == []


def test_the_budget_is_the_larger_of_the_batch_budget_and_the_chunk(served):
    served.points = [point(48, 32, 48)]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=16, chunked_prefill_size=64))
    assert [c[:3] for c in served.calls] == [(48, 32, 48)]


def test_a_runner_naming_no_budget_is_left_alone(served):
    served.points = [point(16, 32, 48)]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=None, chunked_prefill_size=-1))
    warmup.warm_up(SimpleNamespace(model=Model((32, 48))))
    assert served.calls == [] and served.asked == []


def test_the_per_token_fp8_kernels_are_warmed_too(served):
    served.scaled = SimpleNamespace(recorder="scaled recorder")
    served.scaled_points = [point(16, 32, 48), point(16, 100, 200)]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=64))
    assert served.scaled_calls == [(16, 32, 48, None)]
    assert served.calls == []
    assert served.scaled.recorder == "scaled recorder"


def test_export_records_none_of_the_calls(served):
    """The scratch calls run with the dispatcher's recorder taken off, and it comes back, a failing call or not."""
    served.points = [point(16, 32, 48)]
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=64))
    assert served.calls == [(16, 32, 48, None)]
    assert served.dispatcher.recorder == "recorder"


def test_no_served_gemm_is_no_work(served, monkeypatch):
    monkeypatch.setattr(warmup, "gemm_dispatcher", lambda: None)
    warmup.warm_up(runner(Model((32, 48)), max_prefill_tokens=64))
    assert served.calls == [] and served.asked == []


def test_a_failing_warm_up_does_not_fail_the_runners_start(served, monkeypatch, caplog):
    def broken(dispatcher):
        raise RuntimeError("no table")

    monkeypatch.setattr(warmup, "served_gemm_points", broken)
    result = object()
    assert warmup.after_graphs(result, runner(Model((32, 48)), max_prefill_tokens=64)) is None
    assert "were not warmed" in caplog.text
    assert served.dispatcher.recorder == "recorder"
