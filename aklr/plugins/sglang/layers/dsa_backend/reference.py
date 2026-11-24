# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in fp32: the fp8 queries and KV rows decoded
exactly, nothing rounded in between (the kernels' bf16 and fp8 steps are theirs)."""

import torch

from aklr.core.contract.reference import OpReference
from aklr.ops.topk.reference import AnyOrderInFullRowsReference

__all__ = [
    "ROPE_DIM",
    "DsaSparseMlaReference",
    "DsaTopkTransformReference",
]

# The RoPE part of a query head and of a KV row: the DSA models' (sglang's kernel takes a
# power of 2)
ROPE_DIM = 64
# tokens a step: a chunk gathers [tokens, topk, kv_dim] rows
_CHUNK_TOKENS = 16


class DsaSparseMlaReference(OpReference):
    def reference(self, *, q, kv_cache, topk_indices, sm_scale, **_):
        """Decode or prefill: each head's softmax over the scores q . row * sm_scale of the
        rows its token's top-k slots name (-1: none, never a weight), weighing the rows'
        latent parts; 0 for a token that reads no row."""
        latent = q.shape[-1] - ROPE_DIM
        q = q.float()
        # gathered as bytes: the fp8 rows decode only where a token reads them
        pool = kv_cache.view(torch.uint8)
        slots = topk_indices.long()
        output = q.new_zeros((*q.shape[:2], latent))
        for start in range(0, q.shape[0], _CHUNK_TOKENS):
            chunk = slice(start, start + _CHUNK_TOKENS)
            picked = slots[chunk]
            rows = pool[picked.clamp(min=0)].view(kv_cache.dtype).float()
            scores = torch.einsum("thc,tkc->thk", q[chunk], rows) * sm_scale
            scores = scores.masked_fill((picked < 0)[:, None], -torch.inf)
            # a token that reads no row gets 0
            weights = scores.softmax(-1).nan_to_num(0.0)
            output[chunk] = torch.einsum("thk,tkr->thr", weights, rows[..., :latent])
        return {"output": output.to(torch.bfloat16)}


class DsaTopkTransformReference(AnyOrderInFullRowsReference):
    def reference(self, *, score, lengths, page_table_size_1, cu_seqlens_q, out_page_indices, **_):
        """Each row's top-k positions, or all of them in order for a short row, through its
        sequence's page table."""
        topk = out_page_indices.shape[-1]
        width = min(topk, score.shape[-1])
        lengths = lengths.long().unsqueeze(-1)
        positions = torch.arange(score.shape[-1], device=score.device)
        # float32 compares exactly: the scores rank as stored.
        ranked = score.masked_fill(positions >= lengths, -torch.inf).argsort(
            dim=-1, descending=True, stable=True
        )[:, :width]
        selected = torch.where(lengths <= topk, positions[:width], ranked)
        is_selected = positions[:width] < lengths
        rows = torch.arange(score.shape[0], device=score.device)
        sequences = torch.searchsorted(cu_seqlens_q[1:].long(), rows, right=True)
        slots = page_table_size_1[
            sequences[:, None], selected.clamp(max=page_table_size_1.shape[1] - 1)
        ].long()
        out = torch.nn.functional.pad(torch.where(is_selected, slots, -1), (0, topk - width), value=-1)
        return {"out_page_indices": out.to(out_page_indices.dtype)}
