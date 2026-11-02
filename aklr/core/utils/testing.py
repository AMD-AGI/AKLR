###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools
import math
import statistics
import time
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Protocol

import torch

from aklr.core._frozen_model import frozen_dataclass


# shortcut: a hand-made rocprofiler-sdk tool standing in for torch.profiler
@functools.cache
def kernel_tracer():
    """A process must configure() it before its first GPU call: rocprofiler-sdk takes a tool only then."""
    # torch 2.11's Kineto traces ROCm with roctracer and at times records none of a replay's kernels
    from torch.utils.cpp_extension import ROCM_HOME, load

    from aklr.core.platform.env import RuntimeEnv

    build = RuntimeEnv().build_cache_path / "kernel_tracer"
    build.mkdir(parents=True, exist_ok=True)
    tracer = load(
        name="aklr_kernel_tracer",
        sources=[str(Path(__file__).with_name("_kernel_tracer.cpp"))],
        extra_include_paths=[f"{ROCM_HOME}/include"],
        extra_ldflags=[f"-L{ROCM_HOME}/lib", "-lrocprofiler-sdk"],
        build_directory=str(build),
    )
    # warn here, not in configure(): the bench's parent, whose stderr is read, builds; only children configure
    if tracer.rocprofiler_sdk_version >= (1, 4, 2) and torch.__version__ >= (2, 12):
        warnings.warn(  # retire on torch >= 2.12 (Kineto on rocprofiler-sdk) with rocprofiler-sdk >= 1.4.2
            f"torch {torch.__version__}'s Kineto traces with a rocprofiler-sdk that fixed ROCM-29087: trace "
            "bench_cudagraph's replays with torch.profiler and delete aklr/core/utils/_kernel_tracer.cpp",
            FutureWarning,  # Python's default filters hide a DeprecationWarning
            stacklevel=2,
        )
    elif tracer.rocprofiler_sdk_version >= (1, 4, 2):  # rocprofiler-sdk 1.4.2 (ROCm 10.2) fixed ROCM-29087
        warnings.warn(
            "rocprofiler-sdk fixed ROCM-29087: delete the WORKAROUND in aklr/core/utils/_kernel_tracer.cpp",
            FutureWarning,
            stacklevel=2,
        )
    return tracer


class GpuTimer(Protocol):
    """Times a subject on the GPU, one implementation per way of timing, its fields that way's parameters."""

    def measure(self, fn: Callable[[], object], rep: float) -> list[float]: ...


@frozen_dataclass
class CudaGraphTimer(GpuTimer):
    """The GpuTimer of bench_cudagraph; MEASUREMENT_PROFILE records its fields."""

    cache_flush_bytes: int = 1 << 30
    capture_warmup_runs: int = 1
    estimation_samples: int = 5
    # the replays of a one-call graph must agree, and they name the kernels of one call
    single_call_replays: int = 6
    profiler_warmup_replays: int = 1
    measurement_replays: int = 10
    profiler_drain_replays: int = 1
    # the kernel records of all of a measured graph's replays together
    kernel_record_budget: int = 2048

    # a dispatch's record can trail the synchronize that saw its kernel end: 0.1 s seen under load
    record_wait_seconds: ClassVar[float] = 10.0
    record_poll_seconds: ClassVar[float] = 0.001

    def measure(self, fn: Callable[[], object], rep: float) -> list[float]:
        """Per-call GPU ms, flush excluded, one per measured replay of a graph of up to rep ms of calls."""
        if not math.isfinite(rep) or rep <= 0:
            raise ValueError("rep must be a finite positive millisecond budget")

        stream = self._capture_stream(torch.cuda.current_device())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.capture_warmup_runs):
                self.flush_l2_cache()
                fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(self.estimation_samples):
                self.flush_l2_cache()
                fn()
            end_event.record()
            torch.cuda.synchronize()

        estimate_ms = start_event.elapsed_time(end_event) / self.estimation_samples

        # a call is its flush (the graph's root, so its first kernel), then the subject's kernels
        one_call = self._trace(fn, stream, 1, self.single_call_replays, None)
        per_call = Counter(kernel for kernel, _, _ in one_call[0])
        flush = min(one_call[0], key=lambda record: record[1])[0] if one_call[0] else None
        if any(Counter(k for k, _, _ in records) != per_call for records in one_call) or (
            per_call.total() < 2 or per_call[flush] != 1
        ):
            raise RuntimeError(
                f"replays of a single call are not alike flushes and the subject's kernels: {one_call}"
            )

        replays = self.profiler_warmup_replays + self.measurement_replays + self.profiler_drain_replays
        affordable = self.kernel_record_budget // (replays * per_call.total())
        count = max(1, min(int(rep / estimate_ms), affordable))
        by_replay = self._trace(fn, stream, count, replays, replays * count * per_call.total())
        want = Counter({kernel: n * count for kernel, n in per_call.items()})
        first = self.profiler_warmup_replays
        samples = []
        for replay in range(first, first + self.measurement_replays):
            records = by_replay[replay]
            got = Counter(kernel for kernel, _, _ in records)
            if got != want:
                raise RuntimeError(
                    f"replay {replay} of {count} calls ran kernels {dict(got)}, not {dict(want)}"
                )
            if any(end <= start for _, start, end in records):
                raise RuntimeError("rocprofiler-sdk recorded an invalid kernel duration")
            subject_ns = sum(end - start for kernel, start, end in records if kernel != flush)
            samples.append(subject_ns / 1e6 / count)
        return samples

    def flush_l2_cache(self) -> None:
        """Flush L2 and MALL by reading 1 GiB, not writing: dirty lines cost the next kernel a write-back."""
        cache = torch.empty(self.cache_flush_bytes // 4, dtype=torch.int32, device="cuda")
        blocks = math.ceil(cache.numel() / 4096)
        sums = torch.empty(blocks, dtype=torch.int32, device="cuda")
        self._flush_kernel()[(blocks,)](cache, sums, cache.numel(), 4096)

    @staticmethod
    @functools.cache
    def _capture_stream(device: int) -> torch.cuda.Stream:
        """Warm-up and capture share it: per-stream state made only eagerly exists on the capture stream."""
        return torch.cuda.Stream(device=device)

    @staticmethod
    @functools.cache
    def _flush_kernel():
        import triton
        import triton.language as tl

        @triton.jit
        def _flush_l2_cache_kernel(cache, sums, N: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offset = pid * BLOCK + tl.arange(0, BLOCK)
            tl.store(sums + pid, tl.sum(tl.load(cache + offset, offset < N, other=0)))

        return _flush_l2_cache_kernel

    def _trace(
        self,
        fn: Callable[[], object],
        stream: torch.cuda.Stream,
        calls: int,
        replays: int,
        expected: int | None,
    ) -> list[list[tuple[int, int, int]]]:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(calls):
                    self.flush_l2_cache()
                    fn()
        except Exception as error:
            raise RuntimeError(f"CUDA Graph capture failed: {error}") from error

        tracer = kernel_tracer()
        torch.cuda.synchronize()
        tracer.start()
        for _ in range(replays):
            tracer.push()
            graph.replay()
            tracer.pop()
        torch.cuda.synchronize()
        tracer.stop()
        deadline = time.monotonic() + self.record_wait_seconds
        while True:
            by_replay: list[list[tuple[int, int, int]]] = [[] for _ in range(replays)]
            for replay, kernel, start, end in tracer.records():
                by_replay[replay].append((kernel, start, end))
            kernels = [Counter(kernel for kernel, _, _ in records) for records in by_replay]
            n_records = sum(map(len, by_replay))
            if expected is not None:
                is_whole = n_records == expected
            else:
                is_whole = n_records > 0 and kernels.count(kernels[0]) == replays
            if is_whole or time.monotonic() > deadline:
                return by_replay
            time.sleep(self.record_poll_seconds)


def flush_l2_cache() -> None:
    """Flush L2 and MALL by reading 1 GiB, not writing: dirty lines cost the next kernel a write-back."""
    CudaGraphTimer().flush_l2_cache()


def bench_cudagraph(fn, rep: float) -> float:
    """The median per-call GPU time of fn in milliseconds, cache flush excluded."""
    return statistics.median(CudaGraphTimer().measure(fn, rep))
