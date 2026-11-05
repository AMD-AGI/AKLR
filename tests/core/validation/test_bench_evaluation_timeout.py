# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The trial's evaluation time limit grows with the outputs it compares and the ranks."""

from types import SimpleNamespace

from aklr.core.validation.stages._bench import _stage


def test_evaluation_time_limit_grows_with_the_outputs_it_compares():
    small = SimpleNamespace(output_bytes=0, n_ranks=1)
    large = SimpleNamespace(output_bytes=23 << 30, n_ranks=1)
    base = _stage.WORKLOAD_TIMEOUT_SECONDS
    assert _stage._evaluation_timeout_seconds([small, small]) == 2 * base
    assert _stage._evaluation_timeout_seconds([large]) == base + 23 * _stage.EVALUATION_SECONDS_PER_OUTPUT_GIB
    # gemm_n3072_k512's m=342073 workload took about 420 s on a busy node
    assert _stage._evaluation_timeout_seconds([large]) > 2 * 420


def test_evaluation_time_limit_grows_with_the_ranks():
    """Every rank's outputs are compared, each against its own reference."""
    one = SimpleNamespace(output_bytes=1 << 30, n_ranks=1)
    eight = SimpleNamespace(output_bytes=1 << 30, n_ranks=8)
    assert _stage._evaluation_timeout_seconds([eight]) == 8 * _stage._evaluation_timeout_seconds([one])
