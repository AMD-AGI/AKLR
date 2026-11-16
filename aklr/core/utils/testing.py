# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI

from __future__ import annotations

import functools
import gc
import math
import os
import statistics
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@functools.cache
def kernel_tracer():
    """Load AKLR's rocprofiler-sdk collector and graph node attribution fix."""
    from torch.utils.cpp_extension import ROCM_HOME, load

    from aklr.core.platform.env import RuntimeEnv

    build = RuntimeEnv().build_cache_path
    if build is not None:
        build = build / "kernel_tracer"
        build.mkdir(parents=True, exist_ok=True)
    tracer = load(
        name="aklr_kernel_tracer",
        sources=[str(Path(__file__).with_name("_kernel_tracer.cpp"))],
        extra_cflags=["-D__HIP_PLATFORM_AMD__"],
        extra_include_paths=[f"{ROCM_HOME}/include"],
        extra_ldflags=[f"-L{ROCM_HOME}/lib", "-lrocprofiler-sdk", "-lamdhip64"],
        build_directory=None if build is None else str(build),
    )
    # SDK 1.4.0 takes a tool at any time and 1.4.2 fixes queue interposition, which torch 2.12 loads at import
    sdk = tracer.rocprofiler_sdk_version
    deprecated = [f"torch {torch.__version__}"] if torch.__version__ < "2.12" else []
    deprecated += [f"rocprofiler-sdk {'.'.join(map(str, sdk))}"] if sdk < (1, 4, 2) else []
    if deprecated:
        warnings.warn(
            f"aklr kernel tracer: {' and '.join(deprecated)} support is deprecated; "
            "it will require torch >= 2.12 with rocprofiler-sdk >= 1.4.2",
            FutureWarning,
        )
    return tracer


if torch.version.hip is not None:
    # registers as torch initializes the GPU, or now if it has: rocprofiler-sdk intercepts only queues made after
    torch.cuda._lazy_call(kernel_tracer)


@triton.jit
def _flush_l2_cache_kernel(cache, sums, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sums + pid, tl.sum(tl.load(cache + offset, offset < N, other=0)))


def flush_l2_cache() -> None:
    """Flush L2 and MALL by reading 1 GiB: written lines would stay dirty for the next kernel."""
    cache = torch.empty(1_073_741_824 // 4, dtype=torch.int32, device="cuda")
    blocks = triton.cdiv(cache.numel(), 4096)
    sums = torch.empty(blocks, dtype=torch.int32, device="cuda")
    _flush_l2_cache_kernel[(blocks,)](cache, sums, cache.numel(), 4096)


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


@contextmanager
def cuda_graph_without_gc(*args, **kwargs):
    # GC finalizing a Triton kernel unloads its module, which invalidates a capture
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        # thread_local: another thread's GPU calls, on its own device, do not invalidate this capture
        with torch.cuda.graph(*args, capture_error_mode="thread_local", **kwargs) as graph:
            yield graph
    finally:
        if gc_was_enabled:
            gc.enable()


@contextmanager
def _kernel_bench_session():
    from aklr.core.utils import _kernel_profiler as profiler

    with tempfile.TemporaryDirectory(prefix="aklr-bench-") as tmpdir:
        session = profiler.start(os.path.join(tmpdir, "profile"), context="shadow", data="tree")
        try:
            yield profiler, session
        finally:
            if session is not None:
                profiler.finalize(session)


def _collect_scope_times(database, prefix):
    scope_times = []

    def kernel_time_ms(node):
        children = node.get("children", [])
        if len(children) == 0:
            return node.get("metrics", {}).get("time (ns)", 0) / 1e6
        return sum(kernel_time_ms(child) for child in children)

    def visit(node):
        name = node.get("frame", {}).get("name", "")
        if name.startswith(prefix):
            time_ms = kernel_time_ms(node)
            if time_ms > 0:
                scope_times.append((name, time_ms))
            return
        for child in node.get("children", []):
            visit(child)

    for node in database:
        visit(node)
    return [time for _, time in sorted(scope_times)]


def bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="median", *, group=None):
    """fn's kernel ms per call in a CUDA graph, minus cache flushes; with group, the slowest rank's."""
    assert return_mode in ["min", "max", "mean", "median", "all"]
    distributed = group is not None and dist.get_world_size(group) > 1

    kernel_tracer()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # the inputs are written on the caller's stream
    with torch.cuda.stream(stream):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            flush_l2_cache()  # the graph replays it too
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))
        if distributed:
            # equal collectives on every rank; also sets up the communicator before capture
            agreed = torch.tensor([n_repeat], device="cuda")
            dist.all_reduce(agreed, dist.ReduceOp.MIN, group=group)
            n_repeat = int(agreed.item())
            ready = torch.ones(1, device="cuda")

        with _kernel_bench_session() as (profiler, session):
            g = torch.cuda.CUDAGraph()
            scope_prefix = f"aklr.{uuid.uuid4().hex}."
            # on the warmed stream, where fn's per-stream state exists
            with cuda_graph_without_gc(g, stream=stream):
                for i in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    flush_l2_cache()
                    if distributed:
                        dist.all_reduce(ready, group=group)  # the ranks start each call together
                    with profiler.scope(f"{scope_prefix}{i:08d}"):
                        fn()
            torch.cuda.synchronize()
            n_retries = 10
            error = None
            try:
                try:
                    for i in range(n_retries):
                        g.replay()
                    torch.cuda.synchronize()
                finally:
                    profiler.deactivate(session, flushing=True)

                times = [
                    t / n_retries for t in _collect_scope_times(profiler.get_data(session), scope_prefix)
                ]
                if distributed and len(times) != n_repeat:
                    raise RuntimeError(f"bench_cudagraph: {len(times)} of {n_repeat} calls timed")
            except Exception as e:
                if not distributed:
                    raise
                error = e
            del g  # its captured collectives would outlive the call in a raised error's traceback

    if distributed:
        # every rank learns each rank's failure, so all fail together and none waits on a collective
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, None if error is None else str(error), group=group)
        if error is not None:
            raise error
        for rank, message in enumerate(errors):
            if message is not None:
                raise RuntimeError(f"rank {rank}: {message}")
        slowest = torch.tensor(times, dtype=torch.float64, device="cuda")
        dist.all_reduce(slowest, dist.ReduceOp.MAX, group=group)
        times = slowest.tolist()
    return _summarize_statistics(times, quantiles, return_mode)
