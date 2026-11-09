# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile, before a worker serves, the AKLR GEMM kernels vLLM's own warmup does not reach.

An AKLR kernel compiles at its first eager call. vLLM's warmup runs each CUDA graph size
eagerly once; a step with more tokens, or the LM head (which runs outside the graphs), would
first reach its kernels while serving, and stall every request in that step. This runs one
eager dummy step per token count AKLR serves a GEMM at beyond the graph sizes, which also
reaches the MoE Points of those steps, and the LM head at each token count AKLR serves it."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import torch

from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.runtime.dispatch import DispatcherRegistry

__all__ = ["warm_up"]

_LOGGER = logging.getLogger(__name__)


def _served_gemm_tokens() -> dict[tuple[int, int], set[int]]:
    """{(n, k): every m} of the committed GEMM kernels this device serves."""
    dispatcher = DispatcherRegistry.default().dispatchers().get("gemm@1.0")
    if dispatcher is None or dispatcher.table is None:
        return {}
    codec = FastKeyCodec.from_contract(dispatcher.contract)
    tokens = defaultdict(set)
    for key in dispatcher.table.runnables:
        axes = codec.to_point(key).axes
        tokens[(axes["n"], axes["k"])].add(axes["m"])
    return tokens


def warm_up(worker: Any) -> None:
    """``worker``: a ``vllm.v1.worker.gpu_worker.Worker`` after its own warmup. Every rank
    serves the same Points, so every rank runs the same collective steps."""
    from vllm.config import CUDAGraphMode

    tokens = _served_gemm_tokens()
    runner = worker.model_runner
    model = runner.get_model()
    # a pipeline stage without the LM head holds a placeholder without weight
    lm_head = getattr(model, "lm_head", None)
    lm_head_nk = None if getattr(lm_head, "weight", None) is None else tuple(lm_head.weight.shape)
    graph_tokens = worker.vllm_config.compilation_config.max_cudagraph_capture_size or 0
    max_tokens = worker.scheduler_config.max_num_batched_tokens
    steps = sorted(
        {m for nk, ms in tokens.items() if nk != lm_head_nk for m in ms if graph_tokens < m <= max_tokens}
    )
    heads = sorted(tokens.get(lm_head_nk, ()))
    start = time.monotonic()
    for m in steps:
        runner._dummy_run(m, cudagraph_runtime_mode=CUDAGraphMode.NONE, skip_eplb=True, remove_lora=False)
    if heads:
        hidden = torch.zeros(
            max(heads), lm_head_nk[1], dtype=lm_head.weight.dtype, device=lm_head.weight.device
        )
        for m in heads:
            model.compute_logits(hidden[:m])
    torch.cuda.synchronize()
    _LOGGER.warning(
        "aklr warmed %d steps past the graph sizes and %d LM head sizes in %.1f s",
        len(steps),
        len(heads),
        time.monotonic() - start,
    )
