# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A torch.profiler run that opts in gets its GPU events from the kernel tracer: the same kernels Kineto's own
tracer records, each tied to the op that launched it, and a graph replay's to the call that captured it. A run that
does not opt in is Kineto's alone."""

import collections
import json
import multiprocessing
import os
import tempfile
from pathlib import Path

import pytest
import torch


def _opted_in() -> dict:
    from aklr.core.utils.testing import KERNEL_TRACER_CONFIG

    return dict(
        activities=[torch.profiler.ProfilerActivity.CPU],
        experimental_config=torch.profiler._ExperimentalConfig(custom_profiler_config=KERNEL_TRACER_CONFIG),
    )


_KINETOS = dict(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])


def _kernels_of_a_step(options: dict) -> tuple[collections.Counter, dict]:
    w = torch.randn(1024, 1024, device="cuda")

    def step():
        torch.softmax(torch.matmul(w, w) * 2, -1).add_(1)

    step()
    torch.cuda.synchronize()
    with torch.profiler.profile(**options) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    kernels = collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    per_op = {
        op: sum(len(e.kernels) for e in prof.events() if e.name == op)
        for op in ("aten::mm", "aten::mul", "aten::_softmax", "aten::add_")
    }
    return kernels, per_op


def _kinetos_then_ours_then_kinetos() -> list:
    from aklr.core.utils.testing import kernel_tracer

    torch.ones(1, device="cuda")  # the GPU runs before the tracer loads
    kinetos = _kernels_of_a_step(_KINETOS)
    kernel_tracer()
    return [kinetos, _kernels_of_a_step(_opted_in()), _kernels_of_a_step(_KINETOS)]


def _trace_of_a_graph_replay() -> list:
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    x = torch.randn(1 << 20, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.profiler.profile(**_opted_in()) as prof:
        with torch.cuda.stream(stream):
            x.mul_(1.0)
            with torch.cuda.graph(graph):
                x.mul_(2.0)
                x.add_(1.0)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text())["traceEvents"]


def _kernels_of_a_graph_with_a_node_added_by_hand() -> list[str]:
    import ctypes

    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    hip = ctypes.CDLL("libamdhip64.so")
    x = torch.randn(1 << 20, device="cuda")
    stream = torch.cuda.Stream()
    with torch.profiler.profile(**_opted_in()) as prof:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            x.mul_(2.0)
            x.add_(1.0)
        raw = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t(0)
        assert hip.hipGraphGetNodes(raw, None, ctypes.byref(count)) == 0
        nodes = (ctypes.c_void_p * count.value)()
        assert hip.hipGraphGetNodes(raw, nodes, ctypes.byref(count)) == 0
        params = ctypes.create_string_buffer(256)  # hipKernelNodeParams, copied whole
        assert hip.hipGraphKernelNodeGetParams(ctypes.c_void_p(nodes[0]), params) == 0
        added, last = ctypes.c_void_p(), (ctypes.c_void_p * 1)(nodes[count.value - 1])
        assert hip.hipGraphAddKernelNode(ctypes.byref(added), raw, last, ctypes.c_size_t(1), params) == 0
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def _error_of(opt_in_with_cuda: bool) -> str:
    if not opt_in_with_cuda:
        os.environ.pop("ROCP_TOOL_ATTACH", None)  # inherited from a parent that imported the tracer's module
        torch.ones(1, device="cuda")  # the GPU runs before the tracer's module is imported
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    options = _opted_in()
    if opt_in_with_cuda:
        options["activities"].append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(**options):
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
def test_an_opted_in_run_records_kinetos_kernels_and_leaves_kinetos_runs_alone():
    kinetos, ours, kinetos_after = _in_a_fresh_process(_kinetos_then_ours_then_kinetos)
    assert ours == kinetos == kinetos_after
    assert all(n == 3 for n in ours[1].values()), ours[1]


@pytest.mark.gpu
@gpu
def test_a_chrome_trace_ties_kernels_to_their_launches():
    events = _in_a_fresh_process(_trace_of_a_graph_replay)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"]: e for e in events if e.get("cat") == "cuda_runtime"}
    ops = {e["args"]["External id"] for e in events if e.get("cat") == "cpu_op"}
    for kernel in kernels:
        assert kernel["args"]["correlation"] in launches
    replayed = [k for k in kernels if "graph launch" in k["args"]]
    eager = [k for k in kernels if "graph launch" not in k["args"]]
    assert eager and all(k["args"]["External id"] in ops for k in eager)
    assert sorted(k["args"]["graph ordinal"] for k in replayed) == [0, 1]
    assert all(k["args"]["capture correlation"] for k in replayed)
    assert {e["id"] for e in events if e.get("ph") == "f"} <= {e["id"] for e in events if e.get("ph") == "s"}


@pytest.mark.gpu
@gpu
def test_a_graph_node_added_by_hand_replays_unattributed_not_unmatched():
    kernels = _in_a_fresh_process(_kernels_of_a_graph_with_a_node_added_by_hand)
    assert len([k for k in kernels if "MulFunctor" in k]) == 2, kernels


@pytest.mark.gpu
@gpu
def test_an_opted_in_run_refuses_kinetos_gpu_tracer_too():
    assert "ProfilerActivity.CPU only" in _in_a_fresh_process(_error_of, True)


@pytest.mark.gpu
@gpu
def test_a_process_not_initialized_for_the_tracer_refuses_to_profile():
    assert "ROCP_TOOL_ATTACH=1" in _in_a_fresh_process(_error_of, False)
