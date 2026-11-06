# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph timing for ranks that must capture the same number of collective calls."""

import math
import statistics
import uuid

import torch

from aklr.core.utils import testing


@testing.graph_kernel_timing_environment()
def bench_cudagraph(fn, *, rep, group):
    """Use the kernel collector with a capture count agreed by every rank."""
    if not math.isfinite(rep) or rep <= 0:
        raise ValueError("rep must be a finite positive millisecond budget")
    testing.kernel_tracer()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            testing.flush_l2_cache()
            fn()
        end.record()
        torch.cuda.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        count = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))
        agreed = torch.tensor([count], device=stream.device)
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN, group=group)
        count = int(agreed.item())

        with testing._kernel_bench_session() as (profiler, session):
            graph = torch.cuda.CUDAGraph()
            prefix = f"aklr.{uuid.uuid4().hex}."
            with testing.cuda_graph_without_gc(graph, stream=stream):
                for i in range(count):
                    testing.flush_l2_cache()
                    with profiler.scope(f"{prefix}{i:08d}"):
                        fn()
            torch.cuda.synchronize()
            replays = 10
            try:
                for _ in range(replays):
                    graph.replay()
                torch.cuda.synchronize()
            finally:
                profiler.deactivate(session, flushing=True)
            times = [
                time / replays for time in testing._collect_scope_times(profiler.get_data(session), prefix)
            ]
    return statistics.median(times)
