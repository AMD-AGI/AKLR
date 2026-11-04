# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The trial's evaluation time limit grows with the outputs it compares."""

from types import SimpleNamespace

from aklr.core.validation.stages._bench import _stage


def test_evaluation_time_limit_grows_with_the_outputs_it_compares():
    small = SimpleNamespace(output_bytes=0)
    large = SimpleNamespace(output_bytes=23 << 30)
    base = _stage.WORKLOAD_TIMEOUT_SECONDS
    assert _stage._evaluation_timeout_seconds([small, small]) == 2 * base
    assert _stage._evaluation_timeout_seconds([large]) == base + 23 * _stage.EVALUATION_SECONDS_PER_OUTPUT_GIB
    # gemm_n3072_k512's m=342073 workload took about 420 s on a busy node
    assert _stage._evaluation_timeout_seconds([large]) > 2 * 420
