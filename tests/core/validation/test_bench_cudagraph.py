# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The CUDA Graph timer warms a subject on the stream it captures the subject on."""

import pytest
import torch

from aklr.core.utils.testing import bench_cudagraph


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.version.hip is None,
    reason="graph capture and Kineto timing require ROCm PyTorch",
)
def test_a_subject_with_per_stream_state_is_measurable():
    """Stands in for AITER's opus split-K: state it may only create eagerly, per
    stream, must already exist on the stream the timer captures it on."""
    registered: set[int] = set()
    x = torch.ones(1 << 20, device="cuda")

    def subject():
        stream = torch.cuda.current_stream().cuda_stream
        if torch.cuda.is_current_stream_capturing():
            assert stream in registered, "workspace not initialized for this stream"
        else:
            registered.add(stream)
        x.mul_(1.0)

    assert bench_cudagraph(subject, rep=1) > 0
