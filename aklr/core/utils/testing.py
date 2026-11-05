###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools
import math
import statistics
from collections import Counter
from typing import Any

import torch
import triton
import triton.language as tl

KERNEL_RECORD_BUDGET = 2048

TRACE_ROW_RESERVE = 24 * KERNEL_RECORD_BUDGET

_trace_rows_reserved = False


def _reserve_trace_rows() -> None:
    """Grow Kineto's activity list once per process, inside an open session, so a record
    the roctracer backend appends after the stop cannot reallocate the list it walks
    unlocked. Each stream query adds one host-side row and no GPU record."""
    global _trace_rows_reserved
    if _trace_rows_reserved:
        return
    stream = torch.cuda.current_stream()
    for _ in range(TRACE_ROW_RESERVE):
        stream.query()
    _trace_rows_reserved = True


@triton.jit
def _flush_l2_cache_kernel(cache, sums, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sums + pid, tl.sum(tl.load(cache + offset, offset < N, other=0)))


CACHE_FLUSH_BYTES = 1024 * 1024 * 1024


def flush_l2_cache() -> None:
    """Flush L2 and MALL by reading a 1 GiB GPU buffer, not writing it: written lines
    stay dirty, and the next kernel pays their write-back."""
    cache = torch.empty(CACHE_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    blocks = triton.cdiv(cache.numel(), 4096)
    sums = torch.empty(blocks, dtype=torch.int32, device="cuda")
    _flush_l2_cache_kernel[(blocks,)](cache, sums, cache.numel(), 4096)


@functools.cache
def _capture_stream(device: int) -> torch.cuda.Stream:
    """The stream a process both warms subjects on and captures them on, so per-stream
    state a subject may only create eagerly exists on the stream it is captured on."""
    return torch.cuda.Stream(device=device)


# ROCm 7.2 HIP bug (fixed in 10.0): a traced graph launch reads its graph exec's kernel
# names on the HSA signal thread after synchronize returns, so freeing the graph can
# crash (SIGSEGV/SIGABRT in ReportActivity). Keep timed graphs alive, in one shared
# pool. Remove once the CI image ships ROCm >= 10.0.
_kept_graphs: list[torch.cuda.CUDAGraph] = []


@functools.cache
def _graph_pool(device: int) -> tuple[int, int]:
    return torch.cuda.graph_pool_handle()


def _capture(stream: torch.cuda.Stream) -> tuple[torch.cuda.CUDAGraph, Any]:
    """A kept graph and its capture context."""
    graph = torch.cuda.CUDAGraph()
    _kept_graphs.append(graph)
    pool = _graph_pool(stream.device.index)
    return graph, torch.cuda.graph(graph, stream=stream, pool=pool)


def _records_per_call(fn, stream: torch.cuda.Stream):
    graph, capture = _capture(stream)
    with capture:
        flush_l2_cache()
        fn()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    ) as profiler:
        _reserve_trace_rows()
        for replays in (1, 6):
            for _ in range(replays):
                graph.replay()
            torch.cuda.synchronize()
            profiler.step()

    records = Counter()
    for event in profiler.profiler.kineto_results.events():
        if event.device_type() == torch.autograd.DeviceType.CUDA and not event.is_user_annotation():
            records[event.correlation_id()] += 1
    if not records:
        raise RuntimeError("Kineto recorded no kernels for a single call")
    return max(records.values())


def _kineto_replay_times_ms(events, calls_per_graph, replays, flush_name):
    launches = sorted(
        (
            event
            for event in events
            if event.device_type() == torch.autograd.DeviceType.CPU
            and event.name().startswith(("hipGraphLaunch", "cudaGraphLaunch"))
        ),
        key=lambda event: event.start_ns(),
    )
    if len(launches) != replays + 2:
        raise RuntimeError("Kineto did not record every graph launch")
    # Exclude the warmup and drain replays by launch ID: the tracer's trace of them may be incomplete.
    times = dict.fromkeys((event.correlation_id() for event in launches[1 : replays + 1]), 0.0)
    counts = {replay_id: Counter() for replay_id in times}
    for event in events:
        if event.device_type() != torch.autograd.DeviceType.CUDA or event.is_user_annotation():
            continue
        replay_id, name = event.correlation_id(), event.name()
        if replay_id not in times:
            continue
        counts[replay_id][name] += 1
        if name != flush_name:
            ms = event.duration_ns() / 1e6
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("Kineto recorded an invalid kernel duration")
            times[replay_id] += ms
    expected = next(iter(counts.values()))
    for names in counts.values():
        if names.get(flush_name) != calls_per_graph:
            raise RuntimeError(
                f"Incomplete Kineto trace: expected {calls_per_graph} L2 flushes per replay, "
                f"got {names.get(flush_name, 0)}"
            )
        if len(names) == 1:
            raise RuntimeError("Kineto recorded no target kernels")
        if names != expected or any(count % calls_per_graph for count in names.values()):
            raise RuntimeError("Kineto recorded incomplete or inconsistent kernels")
    return [ms / calls_per_graph for ms in times.values()]


def bench_cudagraph(fn, rep: float = 20, group=None) -> float:
    """Return the median per-call GPU time in milliseconds, cache flush excluded; with a
    torch.distributed ``group``, every rank of which times its share of one collective
    ``fn``, each captures as many calls."""
    if not math.isfinite(rep) or rep <= 0:
        raise ValueError("rep must be a finite positive millisecond budget")

    stream = _capture_stream(torch.cuda.current_device())
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # warmup
        flush_l2_cache()
        fn()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            flush_l2_cache()
            fn()
        end_event.record()
        torch.cuda.synchronize()

    estimate_ms = start_event.elapsed_time(end_event) / 5
    affordable = KERNEL_RECORD_BUDGET // (12 * _records_per_call(fn, stream))
    count = max(1, min(int(rep / estimate_ms), affordable))
    if group is not None:
        agreed = torch.tensor([count], device=stream.device)
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN, group=group)
        count = int(agreed.item())

    graph, capture = _capture(stream)
    try:
        with capture:
            for _ in range(count):
                flush_l2_cache()
                fn()
    except Exception as error:
        raise RuntimeError(f"CUDA Graph capture failed: {error}") from error

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    ) as profiler:
        for replays in (1, 12):
            for _ in range(replays):
                graph.replay()
            torch.cuda.synchronize()
            profiler.step()
    samples = _kineto_replay_times_ms(
        profiler.profiler.kineto_results.events(), count, 10, "_flush_l2_cache_kernel"
    )
    return statistics.median(samples)
