# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in fp32: the fp8 queries and KV rows decoded
exactly, nothing rounded in between (the kernels' bf16 and fp8 steps are theirs)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["dsa_sparse_mla_decode_reference", "dsa_topk_transform_reference"]

# tokens a step: a chunk gathers [tokens, topk, kv_dim] rows
_CHUNK_TOKENS = 16


def dsa_sparse_mla_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each head's softmax over the scores q . row * sm_scale of the rows its token's
    top-k slots name (-1: none, never a weight), weighing the rows' latent parts; 0 for a
    token that reads no row."""
    import torch

    args = row_major_args
    latent = point.axes["kv_dim"] - ROPE_DIM
    q = decode(args["q"], point.dtypes["q"]).float()
    # gathered as bytes: the fp8 rows decode only where a token reads them
    pool = args["kv_cache"].view(torch.uint8)
    slots = args["topk_indices"].long()
    output = q.new_zeros((*q.shape[:2], latent))
    for start in range(0, q.shape[0], _CHUNK_TOKENS):
        chunk = slice(start, start + _CHUNK_TOKENS)
        picked = slots[chunk]
        rows = pool[picked.clamp(min=0)].view(args["kv_cache"].dtype)
        rows = decode(rows, point.dtypes["kv_cache"]).float()
        scores = torch.einsum("thc,tkc->thk", q[chunk], rows) * args["sm_scale"]
        scores = scores.masked_fill((picked < 0)[:, None], -torch.inf)
        # a token that reads no row gets 0
        weights = scores.softmax(-1).nan_to_num(0.0)
        output[chunk] = torch.einsum("thk,tkr->thr", weights, rows[..., :latent])
    return output


def dsa_topk_transform_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each row's top-k positions, or all of them in order for a short row, through its
    sequence's page table."""
    import torch

    args = row_major_args
    scores, topk = args["score"], point.axes["topk"]
    width = min(topk, scores.shape[-1])
    lengths = args["lengths"].long().unsqueeze(-1)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    # float32 compares exactly: ranking needs no decode.
    ranked = scores.masked_fill(positions >= lengths, -torch.inf).argsort(
        dim=-1, descending=True, stable=True
    )[:, :width]
    selected = torch.where(lengths <= topk, positions[:width], ranked)
    is_selected = positions[:width] < lengths
    rows = torch.arange(scores.shape[0], device=scores.device)
    sequences = torch.searchsorted(args["cu_seqlens_q"][1:].long(), rows, right=True)
    table = args["page_table_size_1"]
    slots = table[sequences[:, None], selected.clamp(max=table.shape[1] - 1)].long()
    return torch.nn.functional.pad(torch.where(is_selected, slots, -1), (0, topk - width), value=-1).double()
