# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR's GEMM warmup after vLLM's: which served kernels it calls, and that export records none of the calls."""

from types import SimpleNamespace

import pytest
import torch

from aklr.plugins.vllm import warmup

GRAPH_TOKENS, MAX_TOKENS = 8, 64


def point(m, n, k):
    return SimpleNamespace(axes={"m": m, "n": n, "k": k})


class Model(torch.nn.Module):
    def __init__(self, *shapes, lm_head=None):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(s)) for s in shapes])
        if lm_head is not None:
            self.lm_head = SimpleNamespace(weight=torch.zeros(lm_head))


def worker(model, drafter=None):
    runner = SimpleNamespace(get_model=lambda: model)
    if drafter is not None:
        runner.drafter = SimpleNamespace(model=drafter)
    return SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=GRAPH_TOKENS)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_TOKENS),
    )


@pytest.fixture
def served(monkeypatch):
    """A dispatcher serving the test's Points; the Points called, each with the recorder it saw."""
    state = SimpleNamespace(points=[], calls=[], dispatcher=SimpleNamespace(recorder="recorder"))
    monkeypatch.setattr(warmup, "_gemm_dispatcher", lambda: state.dispatcher)
    monkeypatch.setattr(warmup, "_served_gemm_points", lambda dispatcher: state.points)

    def call_once(p, device):
        state.calls.append((p.axes["m"], p.axes["n"], p.axes["k"], state.dispatcher.recorder))
        return True

    monkeypatch.setattr(warmup, "_call_once", call_once)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return state


def test_only_the_rank_models_weight_shapes_past_the_graphs_are_warmed(served):
    served.points = [
        point(16, 32, 48),  # the model's weight, past the graph sizes
        point(16, 48, 32),  # the same weight, the other way round
        point(16, 24, 40),  # the drafter's weight
        point(16, 100, 200),  # no weight of this rank's
        point(4, 32, 48),  # inside the graph sizes: vLLM's warmup reached it
        point(128, 32, 48),  # past the step's token budget
        point(2, 1000, 32),  # the LM head, at any size
    ]
    warmup.warm_up(worker(Model((32, 48), lm_head=(1000, 32)), drafter=Model((24, 40))))
    assert [c[:3] for c in served.calls] == [(16, 32, 48), (16, 48, 32), (16, 24, 40), (2, 1000, 32)]


def test_export_records_no_scratch_call_and_gets_its_recorder_back(served):
    served.points = [point(16, 32, 48)]
    warmup.warm_up(worker(Model((32, 48))))
    assert served.calls == [(16, 32, 48, None)]
    assert served.dispatcher.recorder == "recorder"


def test_the_recorder_comes_back_when_a_call_fails(served, monkeypatch):
    served.points = [point(16, 32, 48)]

    def fails(p, device):
        raise RuntimeError("build failed")

    monkeypatch.setattr(warmup, "_call_once", fails)
    with pytest.raises(RuntimeError):
        warmup.warm_up(worker(Model((32, 48))))
    assert served.dispatcher.recorder == "recorder"
