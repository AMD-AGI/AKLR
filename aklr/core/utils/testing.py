###############################################################################
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Adapted from Triton (https://github.com/triton-lang/triton), MIT License
# Source: python/triton/testing.py@1887644 (_quantile, _summarize_statistics,
#         cuda_graph_without_gc, _collect_proton_scope_times, do_bench_cudagraph_proton)
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
###############################################################################

from __future__ import annotations

import functools
import gc
import math
import statistics
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import triton
import triton.language as tl


# shortcut: no ROCm profiler attributes graph replays yet; retire with ROCm/rocm-systems #14204, #14211
@functools.cache
def kernel_tracer():
    """torch.profiler's GPU tracer in place of Kineto's; load it before the process's first GPU call."""
    from torch.utils.cpp_extension import ROCM_HOME, include_paths, load

    from aklr.core.platform.env import RuntimeEnv

    build = RuntimeEnv().build_cache_path
    if build is not None:
        build = build / "kernel_tracer"
        build.mkdir(parents=True, exist_ok=True)
    tracer = load(
        name="aklr_kernel_tracer",
        sources=[
            str(Path(__file__).with_name(name)) for name in ("_kernel_tracer.cpp", "_kineto_profiler.cpp")
        ],
        # Kineto's headers use fmt, whose compiled half libtorch does not export
        extra_cflags=["-D__HIP_PLATFORM_AMD__", "-DFMT_HEADER_ONLY"],
        extra_include_paths=[f"{ROCM_HOME}/include", str(Path(include_paths()[0]) / "kineto")],
        extra_ldflags=[f"-L{ROCM_HOME}/lib", "-lrocprofiler-sdk", "-lamdhip64"],
        build_directory=None if build is None else str(build),
    )
    # warned by the bench's parent, whose stderr is read and which loads it first
    if tracer.rocprofiler_sdk_version >= (1, 4, 2):  # rocprofiler-sdk 1.4.2 (ROCm 10.2) fixed ROCM-29087
        warnings.warn(
            "rocprofiler-sdk fixed ROCM-29087: delete the WORKAROUND in aklr/core/utils/_kernel_tracer.cpp",
            FutureWarning,  # Python's default filters hide a DeprecationWarning
            stacklevel=2,
        )
    return tracer


@triton.jit
def _flush_l2_cache_kernel(cache, sums, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sums + pid, tl.sum(tl.load(cache + offset, offset < N, other=0)))


CACHE_FLUSH_BYTES = 1024 * 1024 * 1024


def flush_l2_cache() -> None:
    """Flush L2 and MALL by reading 1 GiB: written lines would stay dirty for the next kernel."""
    cache = torch.empty(CACHE_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
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
        with torch.cuda.graph(*args, **kwargs) as graph:
            yield graph
    finally:
        if gc_was_enabled:
            gc.enable()


def _collect_scope_times(events, prefix):
    scope_times = []
    for event in events:
        if event.name.startswith(prefix):
            time_ms = sum(kernel.duration for kernel in event.kernels) / 1e3
            if time_ms > 0:
                scope_times.append((event.name, time_ms))
    return [time for _, time in sorted(scope_times)]


def bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="median"):
    """The per-call GPU time in milliseconds of fn's kernels, replayed in a CUDA graph, cache flushes excluded."""
    assert return_mode in ["min", "max", "mean", "median", "all"]

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

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as session:
            g = torch.cuda.CUDAGraph()
            scope_prefix = f"aklr.{uuid.uuid4().hex}."
            # on the warmed stream, where fn's per-stream state exists
            with cuda_graph_without_gc(g, stream=stream):
                for i in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    flush_l2_cache()
                    with torch.profiler.record_function(f"{scope_prefix}{i:08d}"):
                        fn()
            torch.cuda.synchronize()
            n_retries = 10
            for i in range(n_retries):
                g.replay()
            torch.cuda.synchronize()

        times = [t / n_retries for t in _collect_scope_times(session.events(), scope_prefix)]

    return _summarize_statistics(times, quantiles, return_mode)
