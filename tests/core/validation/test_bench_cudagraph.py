# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public graph timing behavior, with each GPU scenario in a fresh process."""

import ctypes
import math
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _assert_milliseconds(value):
    assert isinstance(value, float)
    assert math.isfinite(value) and value > 0


def _per_stream_state(bench_cudagraph):
    initialized = set()
    x = torch.zeros(1 << 20, device="cuda")

    def subject():
        stream = torch.cuda.current_stream().cuda_stream
        if torch.cuda.is_current_stream_capturing():
            assert stream in initialized, "workspace not initialized for this stream"
        else:
            initialized.add(stream)
        x.fill_(7)

    _assert_milliseconds(bench_cudagraph(subject, rep=1))
    torch.testing.assert_close(x, torch.full_like(x, 7))


def _forked_stream(bench_cudagraph):
    x = torch.zeros(1 << 20, device="cuda")
    y = torch.zeros_like(x)
    side = torch.cuda.Stream()

    def subject():
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            x.fill_(3)
        y.fill_(5)
        current.wait_stream(side)

    _assert_milliseconds(bench_cudagraph(subject))
    torch.testing.assert_close(x, torch.full_like(x, 3))
    torch.testing.assert_close(y, torch.full_like(y, 5))


def _device_copy(bench_cudagraph):
    x = torch.randn(1 << 22, device="cuda")
    y = torch.empty_like(x)

    def subject():
        y.copy_(x)
        y.mul_(2)

    _assert_milliseconds(bench_cudagraph(subject))
    torch.testing.assert_close(y, x * 2)


def _destroy_unused_stream(bench_cudagraph):
    x = torch.zeros(1024, device="cuda")
    hip_path = next(
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libamdhip64.so" in line
    )
    hip = ctypes.CDLL(hip_path)

    def subject():
        stream = ctypes.c_void_p()
        assert hip.hipStreamCreate(ctypes.byref(stream)) == 0
        assert hip.hipStreamDestroy(stream) == 0
        x.fill_(7)

    _assert_milliseconds(bench_cudagraph(subject, rep=1))
    torch.testing.assert_close(x, torch.full_like(x, 7))


def _return_modes_and_quantiles(bench_cudagraph):
    x = torch.zeros(1024, device="cuda")

    def subject():
        x.fill_(7)

    for mode in ("min", "max", "mean", "median"):
        _assert_milliseconds(bench_cudagraph(subject, rep=1, return_mode=mode))
    samples = bench_cudagraph(subject, rep=1, return_mode="all")
    assert isinstance(samples, list) and samples
    for value in samples:
        _assert_milliseconds(value)
    _assert_milliseconds(bench_cudagraph(subject, rep=1, quantiles=[0.5]))
    quantiles = bench_cudagraph(subject, rep=1, quantiles=[0, 0.5, 1])
    assert isinstance(quantiles, list) and len(quantiles) == 3
    assert quantiles == sorted(quantiles)
    for value in quantiles:
        _assert_milliseconds(value)
    with pytest.raises(ValueError, match="Quantiles must be in the range"):
        bench_cudagraph(subject, rep=1, quantiles=[1.1])


def _capture_exception_and_recovery(bench_cudagraph):
    x = torch.zeros(1024, device="cuda")

    def failing_subject():
        x.add_(1)
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("subject failed during capture")

    with pytest.raises(ValueError, match="subject failed during capture"):
        bench_cudagraph(failing_subject, rep=1)
    _assert_milliseconds(bench_cudagraph(lambda: x.fill_(7), rep=1))
    torch.testing.assert_close(x, torch.full_like(x, 7))


def _run_scenario(scenario):
    from aklr.core.utils.testing import bench_cudagraph

    torch.cuda.set_device(0)
    scenario(bench_cudagraph)


@pytest.mark.gpu
@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm PyTorch")
@pytest.mark.parametrize(
    "scenario",
    [
        _per_stream_state,
        _forked_stream,
        _device_copy,
        _destroy_unused_stream,
        _return_modes_and_quantiles,
        _capture_exception_and_recovery,
    ],
    ids=[
        "per_stream_state",
        "forked_stream",
        "device_copy",
        "destroy_unused_stream",
        "return_modes",
        "capture_exception",
    ],
)
def test_bench_cudagraph(scenario):
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        pool.apply_async(_run_scenario, (scenario,)).get(timeout=300)


@pytest.mark.gpu
@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm PyTorch")
def test_bench_cudagraph_refuses_a_late_tracer():
    """Queues made before the tracer registers cannot be intercepted, so attribution refuses them."""
    late = (
        "import torch\n"
        "x = torch.zeros(1024, device='cuda')\n"
        "from aklr.core.utils.testing import bench_cudagraph\n"
        "bench_cudagraph(lambda: x.fill_(7), rep=1)\n"
    )
    result = subprocess.run([sys.executable, "-c", late], capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "before the process's first kernel" in result.stderr


def _group_rank(rank, rendezvous, fault=False):
    import datetime

    import torch.distributed as dist

    from aklr.core.utils.testing import bench_cudagraph, kernel_tracer

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=120)
    )
    try:
        value = torch.zeros(1 << 16, device="cuda")

        def collective():
            value.fill_(rank + 1)
            dist.all_reduce(value)

        if fault:
            if rank == 0:
                kernel_tracer().fail_next_window("rocprofiler-sdk dropped 1 records")
            with pytest.raises(RuntimeError) as error:  # its traceback stays alive until the group ends
                bench_cudagraph(collective, group=dist.group.WORLD)
            return str(error.value)
        # rank 1's budget alone would capture far more calls than rank 0's
        times = bench_cudagraph(
            collective, rep=0 if rank == 0 else 5, group=dist.group.WORLD, return_mode="all"
        )
        torch.testing.assert_close(value, torch.full_like(value, 3))
        return times
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm PyTorch")
def test_bench_cudagraph_group(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    rendezvous = (tmp_path / "store").as_uri()
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        jobs = [pool.apply_async(_group_rank, (rank, rendezvous)) for rank in range(2)]
        rank0, rank1 = (job.get(timeout=300) for job in jobs)
    assert len(rank0) == 1  # the fewest calls any rank's budget allows
    assert rank0 == rank1  # each call reads as its slowest rank's time
    _assert_milliseconds(rank0[0])


@pytest.mark.gpu
@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm PyTorch")
def test_bench_cudagraph_group_fails_together(tmp_path):
    """A rank's tracer fault fails every rank with its message, instead of leaving a peer waiting."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    rendezvous = (tmp_path / "store").as_uri()
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        jobs = [pool.apply_async(_group_rank, (rank, rendezvous, True)) for rank in range(2)]
        rank0, rank1 = (job.get(timeout=200) for job in jobs)
    assert rank0 == "rocprofiler-sdk dropped 1 records"
    assert rank1 == "rank 0: rocprofiler-sdk dropped 1 records"
