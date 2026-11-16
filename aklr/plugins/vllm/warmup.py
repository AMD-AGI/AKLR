# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile, before a worker serves, the AKLR GEMM kernels vLLM's own warmup does not reach.

An AKLR kernel compiles at its first eager call. vLLM's warmup runs each CUDA graph size
eagerly once; a step with more tokens, or the LM head (which runs outside the graphs), would
first reach its kernels while serving, and stall every request in that step. This calls each
of those kernels once on scratch operands. It runs no model step: a dummy step after vLLM's
warmup leaves Kimi-K3 on vLLM's v2 model runner (linear attention, DSpark drafter) wrong at
every decode step after it."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from aklr.core.contract import semantics
from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.ops.gemm.api import gemm

__all__ = ["warm_up"]

_LOGGER = logging.getLogger(__name__)


def _served_gemm_points() -> list[Any]:
    """The Points of the committed GEMM kernels this device serves."""
    dispatcher = DispatcherRegistry.default().dispatchers().get("gemm@1.0")
    if dispatcher is None or dispatcher.table is None:
        return []
    codec = FastKeyCodec.from_contract(dispatcher.contract)
    return [codec.to_point(key) for key in dispatcher.table.runnables]


def _call_once(point: Any, device: torch.device) -> bool:
    """One call of the kernel serving ``point``, on zeros; whether AKLR served it."""

    def zeros(shape, name):
        return torch.zeros(shape, dtype=semantics.torch_dtypes(point.dtypes[name])[0], device=device)

    m, n, k = (point.axes[a] for a in ("m", "n", "k"))
    trans_b = point.attrs["trans_b"]
    result = gemm(
        zeros((m, k), "a"),
        zeros((n, k) if trans_b else (k, n), "b"),
        trans_b=trans_b,
        bias=zeros((n,), "bias") if point.attrs["has_bias"] else None,
        out_dtype=semantics.torch_dtypes(point.dtypes["out"])[0],
    )
    return not isinstance(result, FallbackReason)


def warm_up(worker: Any) -> None:
    """``worker``: a ``vllm.v1.worker.gpu_worker.Worker`` after its own warmup."""
    model = worker.model_runner.get_model()
    # a pipeline stage without the LM head holds a placeholder without weight
    lm_head = getattr(model, "lm_head", None)
    lm_head_nk = None if getattr(lm_head, "weight", None) is None else tuple(lm_head.weight.shape)
    graph_tokens = worker.vllm_config.compilation_config.max_cudagraph_capture_size or 0
    max_tokens = worker.scheduler_config.max_num_batched_tokens
    device = torch.device("cuda", torch.cuda.current_device())
    start = time.monotonic()
    steps = heads = 0
    for point in _served_gemm_points():
        m, nk = point.axes["m"], (point.axes["n"], point.axes["k"])
        if nk == lm_head_nk:
            heads += _call_once(point, device)
        elif graph_tokens < m <= max_tokens:
            steps += _call_once(point, device)
    torch.cuda.synchronize()
    _LOGGER.warning(
        "aklr warmed %d GEMM kernels past the graph sizes and %d LM head sizes in %.1f s",
        steps,
        heads,
        time.monotonic() - start,
    )
