# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The CUDA Graph timer measures the subjects the bench meets, each in a fresh process."""

import multiprocessing

import pytest
import torch


def _time_a_subject_with_per_stream_state() -> float:
    """Stands in for AITER's opus split-K: state it may only create eagerly, per stream, must
    already exist on the stream the timer captures it on."""
    from aklr.core.utils.testing import bench_cudagraph

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
    from aklr.core.utils.testing import bench_cudagraph

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
    from aklr.core.utils.testing import bench_cudagraph

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


class _Event:
    """A Kineto event as torch.profiler's kineto_results gives it."""

    def __init__(self, name, start, end, thread=1, correlation=0, linked=0, gpu=False):
        self._fields = name, start, end, thread, correlation, linked, gpu

    def name(self):
        return self._fields[0]

    def start_ns(self):
        return self._fields[1]

    def end_ns(self):
        return self._fields[2]

    def duration_ns(self):
        return self._fields[2] - self._fields[1]

    def start_thread_id(self):
        return self._fields[3]

    def correlation_id(self):
        return self._fields[4]

    def linked_correlation_id(self):
        return self._fields[5]

    def device_type(self):
        return torch.autograd.DeviceType.CUDA if self._fields[6] else torch.autograd.DeviceType.CPU


def test_a_scope_times_every_kernel_launched_within_it():
    """A kernel linked to an annotation or op the subject opened inside a scope is the scope's; one linked to
    anything outside it, or to another thread's op, is not."""
    from aklr.core.utils.testing import _collect_scope_times

    events = [
        _Event("p.0", 0, 100, correlation=1),
        _Event("inner", 10, 90, correlation=2),
        _Event("aten::add_", 20, 30, correlation=3),
        _Event("p.1", 200, 300, correlation=4),
        _Event("flush", 150, 160, correlation=5),
        _Event("aten::mul_", 210, 220, thread=2, correlation=6),
        _Event("k", 1000, 1_001_000, linked=1, gpu=True),
        _Event("k", 2000, 2_002_000, linked=2, gpu=True),
        _Event("k", 3000, 3_003_000, linked=3, gpu=True),
        _Event("k", 4000, 4_004_000, linked=4, gpu=True),
        _Event("flush", 5000, 9_005_000, linked=5, gpu=True),
        _Event("k", 6000, 9_006_000, linked=6, gpu=True),
    ]
    assert _collect_scope_times(events, "p.") == pytest.approx([6.0, 4.0])
