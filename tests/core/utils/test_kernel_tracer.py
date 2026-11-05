# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A torch.profiler run that opts in gets its GPU events from the kernel tracer: the same kernels Kineto's own
tracer records, each tied to the op that launched it, and a graph replay's to the node and call that captured it,
which calibrating the graph's exec binds without running any of it. A run that does not opt in is Kineto's alone, but
for its GPU tracer, which below HIP 7.16 is refused in a process the tracer ran in."""

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
    ours = _kernels_of_a_step(_opted_in())
    try:
        return [kinetos, ours, _kernels_of_a_step(_KINETOS)]
    except RuntimeError as error:
        return [kinetos, ours, str(error)]


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


def _replayed(prof) -> list[dict]:
    """The replayed kernels of a run, each with its exec, node, grid in work-groups and capturing scope."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    scopes = {e["args"]["External id"]: e["name"] for e in events if e.get("cat") == "user_annotation"}
    return [
        {
            "exec": e["args"]["graph exec"],
            "node": e["args"]["graph node"],
            "groups": e["args"]["grid"][0] // e["args"]["block"][0],
            "scope": scopes.get(e["args"].get("External id")),
        }
        for e in events
        if e.get("cat") == "kernel" and "graph exec" in e["args"]
    ]


def _scopes_of_identical_kernels_on_forked_streams() -> list[tuple[int, str]]:
    """Kernels alike in all but their grid, captured across two streams, which the runtime replays in another order
    than the capture's: each in its own scope, which its grid names."""
    import triton
    import triton.language as tl

    from aklr.core.utils.testing import kernel_tracer

    @triton.jit
    def store(x):
        tl.store(x + tl.program_id(0) * 0, 1.0)

    kernel_tracer()
    x = torch.zeros(1, device="cuda")
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(main):
        store[(1,)](x, num_warps=1)
        torch.cuda.synchronize()
        with torch.profiler.profile(**_opted_in()) as prof:
            with torch.cuda.graph(graph, stream=main):
                for i in range(24):
                    stream = side if i % 3 else main
                    stream.wait_stream(main)
                    with torch.profiler.record_function(f"scope.{i}"), torch.cuda.stream(stream):
                        store[(i + 1,)](x, num_warps=1)
                    if i % 4 == 0:
                        main.wait_stream(side)
                main.wait_stream(side)
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
        kernel_tracer().check()
    return sorted((k["groups"], k["scope"]) for k in _replayed(prof))


def _effects_of_calibrating_a_graph() -> tuple[list[float], list[float]]:
    """The kernel, copy and memset nodes' effects once instantiated, then after three replays."""
    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    count, src = torch.zeros(1, device="cuda"), torch.randn(1 << 16, device="cuda")
    dst, cleared = torch.zeros_like(src), torch.ones_like(src)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    def effects() -> list[float]:
        torch.cuda.synchronize()
        return [count.item(), (dst - src).abs().sum().item(), cleared.sum().item()]

    with torch.cuda.stream(stream):
        count.add_(0)
        torch.cuda.synchronize()
        with torch.profiler.profile(**_opted_in()):
            with torch.cuda.graph(graph, stream=stream):
                count.add_(1)
                dst.copy_(src)
                cleared.zero_()
            instantiated = effects()
            for _ in range(3):
                graph.replay()
            replayed = effects()
        kernel_tracer().check()
    return instantiated, replayed


def _scopes_of_an_exec_that_changes() -> tuple[collections.Counter, int, float]:
    """An exec replayed whole, with a node disabled, after its kernel arguments move, and beside a second exec of
    its graph: each replayed kernel's scope, the execs, and the work done."""
    import ctypes

    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    hip = ctypes.CDLL("libamdhip64.so")
    x = torch.zeros(1024, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(stream):
        x.add_(0)
        torch.cuda.synchronize()
        with torch.profiler.profile(**_opted_in()) as prof:
            with torch.cuda.graph(graph, stream=stream):
                for i in range(3):
                    with torch.profiler.record_function(f"scope.{i}"):
                        x.add_(1.0)
            graph.instantiate()
            raw, exec_ = ctypes.c_void_p(graph.raw_cuda_graph()), ctypes.c_void_p(graph.raw_cuda_graph_exec())
            count = ctypes.c_size_t(0)
            assert hip.hipGraphGetNodes(raw, None, ctypes.byref(count)) == 0
            nodes = (ctypes.c_void_p * count.value)()
            assert hip.hipGraphGetNodes(raw, nodes, ctypes.byref(count)) == 0
            nodes = [ctypes.c_void_p(node) for node in nodes]
            graph.replay()
            assert hip.hipGraphNodeSetEnabled(exec_, nodes[0], 0) == 0
            graph.replay()
            assert hip.hipGraphNodeSetEnabled(exec_, nodes[0], 1) == 0
            params = ctypes.create_string_buffer(256)  # hipKernelNodeParams, copied whole
            assert hip.hipGraphKernelNodeGetParams(nodes[1], params) == 0
            assert hip.hipGraphExecKernelNodeSetParams(exec_, nodes[1], params) == 0
            graph.replay()
            again = ctypes.c_void_p()
            assert hip.hipGraphInstantiate(ctypes.byref(again), raw, None, None, ctypes.c_size_t(0)) == 0
            assert hip.hipGraphLaunch(again, ctypes.c_void_p(stream.cuda_stream)) == 0
            torch.cuda.synchronize()
            assert hip.hipGraphExecDestroy(again) == 0
        kernel_tracer().check()
    replayed = _replayed(prof)
    return collections.Counter(k["scope"] for k in replayed), len({k["exec"] for k in replayed}), x[0].item()


def _errors_of_graphs_calibration_refuses() -> tuple[list[str], int, list[float]]:
    """Graphs that calibrating would run part of, each launched once inside a window, and one instantiated before
    it: what each run's tracing failed with, how often the host node ran, and where the pinned copy had landed
    after instantiation and after the launch."""
    import ctypes

    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    hip = ctypes.CDLL("libamdhip64.so")
    x = torch.zeros(1024, device="cuda")
    pinned, landed = torch.ones(1024).pin_memory(), torch.zeros(1024, device="cuda")
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    ran = [0]
    on_host = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: ran.__setitem__(0, ran[0] + 1))
    host = (ctypes.c_void_p * 2)(ctypes.cast(on_host, ctypes.c_void_p), None)  # hipHostNodeParams
    errors, copied = [], []

    def failure() -> str:
        try:
            kernel_tracer().check()
        except RuntimeError as error:
            return str(error)
        return ""

    with torch.cuda.stream(stream):
        x.add_(0)
        child = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(child, stream=stream):
            x.add_(1.0)
        for build in ("host", "child"):
            graph, node, exec_ = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            assert hip.hipGraphCreate(ctypes.byref(graph), 0) == 0
            if build == "host":
                assert hip.hipGraphAddHostNode(ctypes.byref(node), graph, None, ctypes.c_size_t(0), host) == 0
            else:
                raw = ctypes.c_void_p(child.raw_cuda_graph())
                assert (
                    hip.hipGraphAddChildGraphNode(ctypes.byref(node), graph, None, ctypes.c_size_t(0), raw)
                    == 0
                )
            with torch.profiler.profile(**_opted_in()):
                assert (
                    hip.hipGraphInstantiate(ctypes.byref(exec_), graph, None, None, ctypes.c_size_t(0)) == 0
                )
                assert hip.hipGraphLaunch(exec_, handle) == 0
                torch.cuda.synchronize()
                assert hip.hipGraphExecDestroy(exec_) == 0
            errors.append(failure())
            assert hip.hipGraphDestroy(graph) == 0
        copy = torch.cuda.CUDAGraph()
        with torch.profiler.profile(**_opted_in()):
            with torch.cuda.graph(copy, stream=stream):
                landed.copy_(pinned, non_blocking=True)
            torch.cuda.synchronize()
            copied.append(landed.sum().item())
            copy.replay()
            torch.cuda.synchronize()
            copied.append(landed.sum().item())
        errors.append(failure())
        before = torch.cuda.CUDAGraph()
        with torch.cuda.graph(before, stream=stream):
            x.add_(1.0)
        with torch.profiler.profile(**_opted_in()):
            before.replay()
            torch.cuda.synchronize()
        errors.append(failure())
    return errors, ran[0], copied


def _scope_of_a_node_added_by_hand_to_a_capture() -> list[str]:
    """A kernel node added by hand to a graph under capture in one scope, which a call in a later scope then makes
    its stream's dependency: the scope each replayed kernel was captured in, in replay order of nodes."""
    import ctypes

    from aklr.core.utils.testing import kernel_tracer

    kernel_tracer()
    hip = ctypes.CDLL("libamdhip64.so")
    x = torch.zeros(1024, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        x.add_(0)
        torch.cuda.synchronize()
        with torch.profiler.profile(**_opted_in()) as prof:
            with torch.cuda.graph(graph, stream=stream):
                with torch.profiler.record_function("scope.a"):
                    x.add_(1.0)
                    status, raw = ctypes.c_int(), ctypes.c_void_p()
                    deps, count = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
                    assert (
                        hip.hipStreamGetCaptureInfo_v2(
                            ctypes.c_void_p(stream.cuda_stream),
                            ctypes.byref(status),
                            None,
                            ctypes.byref(raw),
                            ctypes.byref(deps),
                            ctypes.byref(count),
                        )
                        == 0
                    )
                    params = ctypes.create_string_buffer(256)  # hipKernelNodeParams, copied whole
                    assert hip.hipGraphKernelNodeGetParams(ctypes.c_void_p(deps[0]), params) == 0
                    added = ctypes.c_void_p()
                    assert hip.hipGraphAddKernelNode(ctypes.byref(added), raw, deps, count, params) == 0
                with torch.profiler.record_function("scope.b"):
                    assert (
                        hip.hipStreamUpdateCaptureDependencies(
                            ctypes.c_void_p(stream.cuda_stream), ctypes.byref(added), ctypes.c_size_t(1), 1
                        )
                        == 0
                    )  # hipStreamSetCaptureDependencies
                    x.add_(1.0)
            graph.replay()
            torch.cuda.synchronize()
        kernel_tracer().check()
    return [k["scope"] for k in sorted(_replayed(prof), key=lambda k: k["node"])]


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
def test_an_opted_in_run_records_kinetos_kernels_and_refuses_kinetos_gpu_tracer_after_it():
    kinetos, ours, kinetos_after = _in_a_fresh_process(_kinetos_then_ours_then_kinetos)
    assert ours == kinetos
    assert all(n == 3 for n in ours[1].values()), ours[1]
    assert "trace the GPU with Kineto in another process" in kinetos_after


@pytest.mark.gpu
@gpu
def test_a_chrome_trace_ties_kernels_to_their_launches():
    events = _in_a_fresh_process(_trace_of_a_graph_replay)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"]: e for e in events if e.get("cat") == "cuda_runtime"}
    ops = {e["args"]["External id"] for e in events if e.get("cat") == "cpu_op"}
    for kernel in kernels:
        assert kernel["args"]["correlation"] in launches
    replayed = [k for k in kernels if "graph exec" in k["args"]]
    eager = [k for k in kernels if "graph exec" not in k["args"]]
    assert eager and all(k["args"]["External id"] in ops for k in eager)
    assert sorted(k["args"]["graph node"] for k in replayed) == [0, 1]
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


@pytest.mark.gpu
@gpu
def test_identical_kernels_replayed_out_of_capture_order_keep_their_scopes():
    replayed = _in_a_fresh_process(_scopes_of_identical_kernels_on_forked_streams)
    assert replayed == sorted([(i + 1, f"scope.{i}") for i in range(24)] * 2)


@pytest.mark.gpu
@gpu
def test_calibrating_a_graph_runs_none_of_it():
    instantiated, replayed = _in_a_fresh_process(_effects_of_calibrating_a_graph)
    assert instantiated == [0.0, pytest.approx(instantiated[1]), 65536.0] and instantiated[1] > 0
    assert replayed == [3.0, 0.0, 0.0]


@pytest.mark.gpu
@gpu
def test_an_exec_that_changes_keeps_its_scopes():
    scopes, execs, work = _in_a_fresh_process(_scopes_of_an_exec_that_changes)
    assert scopes == {"scope.0": 3, "scope.1": 4, "scope.2": 4}
    assert execs == 3  # the exec, bound again after its arguments moved, and the second exec
    assert work == 11.0


@pytest.mark.gpu
@gpu
def test_a_graph_calibration_would_run_part_of_is_refused_and_left_to_run_once():
    errors, host_runs, copied = _in_a_fresh_process(_errors_of_graphs_calibration_refuses)
    assert "a node of type 3" in errors[0]
    assert "a child graph node" in errors[1]
    assert "may copy by DMA" in errors[2]
    assert "instantiated outside the profiling window" in errors[3]
    assert host_runs == 1  # the launch's, none of calibration's
    assert copied == [0.0, 1024.0]


@pytest.mark.gpu
@gpu
def test_a_node_added_by_hand_to_a_capture_keeps_the_scope_it_was_added_in():
    assert _in_a_fresh_process(_scope_of_a_node_added_by_hand_to_a_capture) == [
        "scope.a",
        "scope.a",
        "scope.b",
    ]
