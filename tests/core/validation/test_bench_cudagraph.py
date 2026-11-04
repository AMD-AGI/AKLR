# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The CUDA Graph timer measures the subjects the bench meets, each in a fresh process, as the
bench runs it: the timer's tracer is loaded before the first GPU call. A timing is only
returned when every replayed kernel matched the call that captured it."""

import multiprocessing

import pytest
import torch


def _time_a_subject_with_per_stream_state() -> float:
    """Stands in for AITER's opus split-K: state it may only create eagerly, per stream, must
    already exist on the stream the timer captures it on."""
    from aklr.core.utils.testing import bench_cudagraph, kernel_tracer

    kernel_tracer()
    registered: set[int] = set()
    x = torch.ones(1 << 20, device="cuda")

    def subject():
        stream = torch.cuda.current_stream().cuda_stream
        if torch.cuda.is_current_stream_capturing():
            assert stream in registered, "workspace not initialized for this stream"
        else:
            registered.add(stream)
        x.mul_(1.0)

    return bench_cudagraph(subject, rep=1)


def _time_a_subject_that_forks_a_stream() -> float:
    """A call that forks onto a second stream and joins back, at the bench's budget: under
    rocprofiler-sdk 1.3.5's queue interposition, its graph deadlocks (ROCM-29087), and the
    runtime replays one branch before the other."""
    from aklr.core.utils.testing import bench_cudagraph, kernel_tracer

    kernel_tracer()
    x, y = torch.ones(1 << 20, device="cuda"), torch.ones(1 << 20, device="cuda")
    side = torch.cuda.Stream()

    def subject():
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            x.mul_(1.0)
        y.mul_(1.0)
        current.wait_stream(side)

    return bench_cudagraph(subject, rep=20)


def _time_a_subject_that_copies() -> float:
    """A device copy is a memcpy node, which the runtime replays as its own blit kernel."""
    from aklr.core.utils.testing import bench_cudagraph, kernel_tracer

    kernel_tracer()
    x = torch.randn(1 << 22, device="cuda")
    y = torch.empty_like(x)

    def subject():
        y.copy_(x)
        y.mul_(2.0)

    return bench_cudagraph(subject, rep=20)


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.version.hip is None,
    reason="graph capture and rocprofiler-sdk timing require ROCm PyTorch",
)
@pytest.mark.parametrize(
    "time_a_subject",
    [_time_a_subject_with_per_stream_state, _time_a_subject_that_forks_a_stream, _time_a_subject_that_copies],
    ids=["per_stream_state", "forked_stream", "copy"],
)
def test_a_subject_is_measurable(time_a_subject):
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        # a deadlocked timer fails the test, not the suite
        assert pool.apply_async(time_a_subject).get(timeout=300) > 0
