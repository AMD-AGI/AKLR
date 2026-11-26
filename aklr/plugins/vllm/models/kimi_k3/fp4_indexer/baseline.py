# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What Kimi-K3.1's vLLM (0.0.0+mum.14d1ec2336) runs for one query tile of prefill_indexer: its Triton query
quantization (copied in ``_triton``: the fork's file is in no vLLM release), aiter's FlyDSL scorer on the staged pages
with vLLM's page table and schedule, aiter's top-k (unstable, as vLLM calls it under DCP) into rows of -1 and the ids
made relative, in the order vLLM leaves them. aiter 0.1.28 (vLLM 0.34.0's image) has both aiter functions, as the
fork's image does."""

from __future__ import annotations

from aklr.core.contract.baseline import OpBaseline

__all__ = ["KimiK3Fp4IndexerPrefillBaseline"]

# vLLM's _prepare_chunk: the scorer's block of keys, and the persistent grid's least width
_BLOCK_K = 256
_MIN_CTAS = 4096


class KimiK3Fp4IndexerPrefillBaseline(OpBaseline):
    def baseline(
        self,
        *,
        q,
        weights,
        key_values,
        key_scales,
        row_starts,
        row_ends,
        indices,
        weight_scale=0.015625,
    ):
        """indices, written in place as vLLM writes its tile of local_out."""
        import torch
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
            flydsl_pa_mqa_logits_fp4_prefill,
        )
        from aiter.ops.topk import top_k_per_row_prefill

        from aklr.plugins.vllm.models.kimi_k3.fp4_indexer._triton import quantize_queries

        rows, pages = q.shape[0], key_values.shape[0]
        width = pages * 64
        # _prepare_chunk's page table (the staged pages in order; the scorer prefetches one block past the last)
        table = torch.zeros((1, -(-pages // 4) * 4 + 4), device=q.device, dtype=torch.int32)
        table[0, :pages] = torch.arange(pages, device=q.device, dtype=torch.int32)
        batches = torch.zeros(rows, device=q.device, dtype=torch.int32)
        _, schedule, n_ctas = compute_prefill_schedule(
            batches, row_starts, row_ends, _BLOCK_K, max(_MIN_CTAS, rows), width
        )
        qp, qs = quantize_queries(q)
        logits = torch.empty((rows, width), device=q.device, dtype=torch.float32)
        flydsl_pa_mqa_logits_fp4_prefill(
            qp,
            qs,
            key_values.view(torch.uint8).view(pages, 1, 4, 64, 16),
            key_scales.view(torch.uint8).view(pages, 1, 4, 64),
            table,
            weights,
            batches,
            row_starts,
            row_ends,
            width,
            weight_scale=weight_scale,
            cta_info=schedule,
            n_ctas=n_ctas,
            out=logits,
        )
        # vLLM's local_out starts as -1, then its top-k and the ids made relative
        indices.fill_(-1)
        top_k_per_row_prefill(
            logits=logits,
            rowStarts=row_starts,
            rowEnds=row_ends,
            indices=indices,
            values=None,
            numRows=rows,
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            stable=False,
            k=indices.shape[1],
        )
        indices.copy_(torch.where(indices >= 0, indices - row_starts[:, None], -1))
        return {"indices": indices}
