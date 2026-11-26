# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.fp4_indexer.contract import KIMI_K3_FP4_INDEXER_PREFILL

__all__ = ["kimi_k3_fp4_indexer_prefill"]


@api(KIMI_K3_FP4_INDEXER_PREFILL)
def kimi_k3_fp4_indexer_prefill(
    q,
    weights,
    key_values,
    key_scales,
    row_starts,
    row_ends,
    indices,
    weight_scale=0.015625,
): ...


@vllm_route(
    "vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill.prefill_chunks",
    "vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill.prefill_indexer",
    versions=("0.0.0+mum.14d1ec2336",),
)
def _prefill_indexer(
    q,
    weights,
    cache,
    out,
    chunk,
    world,
    interleave,
    weight_scale=1.0,
    staged_keys=None,
    pending=None,
    plan=None,
):
    """A prefill chunk's query tiles of this rank (vLLM's whole prefill_indexer: quantization, scores, top-k, the ids
    made relative), each tile one call, the rows left to the gather of prefill_chunks."""
    # prefill_chunks hands every chunk its staged keys, its plan and the list it gathers the ranks' rows from: a call
    # without them stages or gathers itself (DCP collectives). Without DCP each rank scores every row and vLLM picks
    # them with the stable top-k, the same ids on every rank, which the contract does not promise.
    if staged_keys is None or pending is None or plan is None or world == 1:
        return FallbackReason.OUT_OF_CONTRACT
    payload, scales = staged_keys
    # the staged pages without their one-tile axis: [pages, 4, 64, 16] values, [pages, 4, 64] scales
    key_values, key_scales = payload[:, 0], scales[:, 0]
    local_out = torch.full((plan.per_rank, out.shape[1]), -1, dtype=out.dtype, device=out.device)
    for query in plan.queries:
        start, end = query.start, query.end
        # a tile's row bounds are a slice of the chunk's from this rank's first row: AKLR takes 16-byte aligned
        # tensors, so bounds from an offset not a multiple of 4 rows (61 rows a rank: ranks 1-3, 5-7) are copied
        starts, ends = (
            b if b.data_ptr() % 16 == 0 else b.clone() for b in (query.row_starts, query.row_ends)
        )
        served = kimi_k3_fp4_indexer_prefill(
            q[start:end],
            weights[start:end],
            key_values,
            key_scales,
            starts,
            ends,
            local_out[start - plan.first : end - plan.first],
            weight_scale=weight_scale,
        )
        # a tile the op does not serve: vLLM scores the whole chunk again, local_out dropped
        if isinstance(served, FallbackReason):
            return served
    pending.append((chunk.token_start, chunk.token_end, local_out))
    return None
