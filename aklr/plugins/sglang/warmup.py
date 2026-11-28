# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile, before a worker serves, the AKLR GEMM kernels (16-bit and per-token fp8) sglang's graph capture does
not reach.

An AKLR kernel compiles at its first eager call. sglang's graph capture runs each decode batch size once; a
prefill (a step no graph replays, of any token count up to the server's prefill budget) would first reach its
kernels while serving and stall every request of that step: a second or more at the first batch of two prompts.
After a runner's graphs are captured this calls each GEMM kernel the device serves once on scratch operands, at
each m a workload holds up to the prefill budget, for the weight shapes the runner's model holds
(``aklr.plugins.aiter.provider.gemm.warmup``): no other kernel is built, and no scratch call is recorded as a
workload. A warm-up that cannot run leaves the kernels to their first
call."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from aklr.plugins.aiter.provider.gemm.warmup import (
    call_once,
    call_once_a8w8,
    gemm_a8w8_dispatcher,
    gemm_dispatcher,
    served_gemm_points,
    weight_shapes,
)

__all__ = ["TARGET", "VERSIONS", "after_graphs", "warm_up"]

_LOGGER = logging.getLogger(__name__)

TARGET = "sglang.srt.model_executor.model_runner.ModelRunner.init_cuda_graphs"
# the sglang releases whose ModelRunner this reads (server_args, model)
VERSIONS = ("0.5.23",)


def _prefill_budget(runner: Any) -> int:
    """The most tokens one prefill step of ``runner`` computes: the larger of the server's batch budget and its
    chunk size (0 where the runner names neither)."""
    args = getattr(runner, "server_args", None)
    sizes = [getattr(args, name, None) for name in ("max_prefill_tokens", "chunked_prefill_size")]
    return max([int(size) for size in sizes if isinstance(size, int) and size > 0], default=0)


def warm_up(runner: Any) -> None:
    """``runner``: a ``sglang.srt.model_executor.model_runner.ModelRunner`` after its graphs are captured."""
    upto = _prefill_budget(runner)
    dispatchers = ((gemm_dispatcher(), call_once), (gemm_a8w8_dispatcher(), call_once_a8w8))
    if not upto or all(served is None for served, _ in dispatchers):
        return
    shapes = weight_shapes([getattr(runner, "model", None)])
    device = torch.device("cuda", torch.cuda.current_device())
    start = time.monotonic()
    called = [0, 0]
    for which, (served, call) in enumerate(dispatchers):
        if served is None:
            continue
        # the scratch calls are no workload: export records none of them
        recorder, served.recorder = served.recorder, None
        try:
            for point in served_gemm_points(served):
                # aiter's GEMMs take the weight as [n, k] alone
                if not point.attrs["trans_b"]:
                    continue
                if point.axes["m"] <= upto and (point.axes["n"], point.axes["k"]) in shapes:
                    called[which] += call(point, device)
        finally:
            served.recorder = recorder
    torch.cuda.synchronize()
    _LOGGER.warning(
        "aklr warmed %d GEMM kernels and %d per-token fp8 ones up to %d rows in %.1f s",
        *called,
        upto,
        time.monotonic() - start,
    )


def after_graphs(result: Any, runner: Any, *args: Any, **kwargs: Any) -> None:
    """sglang's AFTER hook on ``TARGET``: the warm-up, which never fails the runner's start."""
    try:
        warm_up(runner)
    except Exception as error:  # noqa: BLE001 (a warm-up that cannot run leaves the kernels to their first call)
        _LOGGER.warning("aklr: the GEMM kernels were not warmed: %r", error)
