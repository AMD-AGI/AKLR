# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile, before a worker serves, the AKLR GEMM kernels vLLM's own warmup does not reach.

An AKLR kernel compiles at its first eager call. vLLM's warmup runs each CUDA graph size
eagerly once; a step with more tokens, or the LM head (which runs outside the graphs), would
first reach its kernels while serving, and stall every request in that step. This calls each
of those kernels once on scratch operands, for the weight shapes the worker's model and drafter
hold (the dispatch table is the device's, any model's): no other kernel is built, and no scratch
call is recorded as a workload. It runs no model step: a dummy step after vLLM's warmup leaves
Kimi-K3 on vLLM's v2 model runner (linear attention, DSpark drafter) wrong at every decode step
after it."""

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


def _gemm_dispatcher() -> Any:
    """The gemm@1.0 dispatcher, None where it serves nothing."""
    dispatcher = DispatcherRegistry.default().dispatchers().get("gemm@1.0")
    return None if dispatcher is None or dispatcher.table is None else dispatcher


def _served_gemm_points(dispatcher: Any) -> list[Any]:
    """The Points of the committed GEMM kernels this device serves."""
    codec = FastKeyCodec.from_contract(dispatcher.contract)
    return [codec.to_point(key) for key in dispatcher.table.runnables]


def _weight_shapes(worker: Any) -> set[tuple[int, int]]:
    """Every 2-D weight shape, either way round, of the worker's model and its speculative drafter's: the (n, k) a
    GEMM of this rank can take."""
    runner = worker.model_runner
    models = [runner.get_model(), getattr(getattr(runner, "drafter", None), "model", None)]
    shapes = set()
    for model in models:
        if isinstance(model, torch.nn.Module):
            for tensor in [*model.parameters(), *model.buffers()]:
                if tensor.dim() == 2:
                    shapes |= {tuple(tensor.shape), tuple(tensor.shape)[::-1]}
    return shapes


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
    dispatcher = _gemm_dispatcher()
    if dispatcher is None:
        return
    shapes = _weight_shapes(worker)
    start = time.monotonic()
    steps = heads = 0
    # the scratch calls are no workload: export records none of them
    recorder, dispatcher.recorder = dispatcher.recorder, None
    try:
        for point in _served_gemm_points(dispatcher):
            m, nk = point.axes["m"], (point.axes["n"], point.axes["k"])
            if nk == lm_head_nk:
                heads += _call_once(point, device)
            elif graph_tokens < m <= max_tokens and nk in shapes:
                steps += _call_once(point, device)
    finally:
        dispatcher.recorder = recorder
    torch.cuda.synchronize()
    _LOGGER.warning(
        "aklr warmed %d GEMM kernels past the graph sizes and %d LM head sizes in %.1f s",
        steps,
        heads,
        time.monotonic() - start,
    )
