# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Loaded before a process's first GPU call, the kernel tracer is torch.profiler's GPU tracer: the same kernels
Kineto's records, each tied to the op that launched it, and a graph replay's to the call that captured it."""

import collections
import json
import multiprocessing
import tempfile
from pathlib import Path

import pytest
import torch


def _eager_kernels(use_tracer: bool) -> collections.Counter:
    if use_tracer:
        from aklr.core.utils.testing import kernel_tracer

        kernel_tracer()
    from torch.profiler import ProfilerActivity, profile

    w = torch.randn(1024, 1024, device="cuda")

    def step():
        torch.softmax(torch.matmul(w, w) * 2, -1).add_(1)

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    assert prof.key_averages().total_average().self_device_time_total > 0
    for op in ("aten::mm", "aten::mul", "aten::_softmax", "aten::add_"):
        assert sum(e.self_device_time_total for e in prof.events() if e.name == op) > 0, op
    return collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )


def _trace_of_a_graph_replay() -> list:
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(1 << 20, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x.mul_(1.0)
        with torch.cuda.graph(graph):
            x.mul_(2.0)
            x.add_(1.0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        x.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text())["traceEvents"]


def _breakdown_of_a_call() -> dict:
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    from aklr.benchmarks.models.kernel_breakdown import profile_call

    w = torch.randn(1024, 1024, device="cuda")
    return profile_call(lambda: torch.matmul(w, w).relu_())


def _profile_after_the_first_gpu_call() -> str:
    torch.ones(1, device="cuda")
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    from torch.profiler import ProfilerActivity, profile

    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            torch.ones(1, device="cuda")
    except RuntimeError as error:
        return str(error)
    return ""


def _in_a_fresh_process(fn, *args):
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply_async(fn, args).get(timeout=300)


gpu = pytest.mark.skipif(torch.version.hip is None, reason="rocprofiler-sdk tracing requires ROCm PyTorch")


@pytest.mark.gpu
@gpu
def test_the_same_kernels_as_kinetos():
    assert _in_a_fresh_process(_eager_kernels, True) == _in_a_fresh_process(_eager_kernels, False)


@pytest.mark.gpu
@gpu
def test_a_chrome_trace_ties_kernels_to_their_launches():
    events = _in_a_fresh_process(_trace_of_a_graph_replay)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"]: e for e in events if e.get("cat") == "cuda_runtime"}
    ops = {e["args"]["External id"] for e in events if e.get("cat") == "cpu_op"}
    assert len(kernels) == 3
    for kernel in kernels:
        assert kernel["args"]["correlation"] in launches
    eager, *replayed = sorted(kernels, key=lambda e: e["ts"])
    assert eager["args"]["External id"] in ops
    assert "graph launch" not in eager["args"]
    assert [k["args"]["graph ordinal"] for k in replayed] == [0, 1]
    assert all(k["args"]["capture correlation"] for k in replayed)
    assert {e["id"] for e in events if e.get("ph") == "f"} <= {e["id"] for e in events if e.get("ph") == "s"}


@pytest.mark.gpu
@gpu
def test_the_kernel_breakdown_reads_its_trace():
    breakdown = _in_a_fresh_process(_breakdown_of_a_call)
    assert len(breakdown) == 2
    assert all(ms > 0 and calls == 1 for ms, calls, _ in breakdown.values())


@pytest.mark.gpu
@gpu
def test_a_tracer_loaded_late_refuses_to_profile():
    assert "before the process's first GPU call" in _in_a_fresh_process(_profile_after_the_first_gpu_call)
