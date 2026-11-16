# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in the dtype ``decode`` returns: the fp8_ds_mla rows decoded
exactly (fp8 codes times their group's scale, the bf16 RoPE values), nothing rounded in between."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["dsa_sparse_mla_dcp_partial_reference"]

# rows a step: a chunk gathers [rows, topk, 656] bytes and decodes them to [rows, topk, 576]
_CHUNK_ROWS = 16
_NOPE, _GROUP, _SCALES = 512, 128, 528


def dsa_sparse_mla_dcp_partial_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """Each head's softmax over the scores q . key * sm_scale of the keys its row's first
    topk_length slots name, weighing the keys' NoPE values, and the natural log of the sum of
    exp(score); 0 and -inf for a row that reads no key."""
    import torch

    args = row_major_args
    q_nope = decode(args["q_nope"], point.dtypes["q_nope"])
    q = torch.cat((q_nope, decode(args["q_rope"], point.dtypes["q_rope"]).to(q_nope.dtype)), dim=-1)
    pool = args["kv_cache"].view(torch.uint8)
    slots = args["indices"].long()
    positions = torch.arange(slots.shape[-1], device=slots.device)
    is_read = positions < args["topk_length"].long()[:, None]
    output = q.new_zeros((*q.shape[:2], _NOPE))
    lse = q.new_empty(q.shape[:2])
    for start in range(0, q.shape[0], _CHUNK_ROWS):
        chunk = slice(start, start + _CHUNK_ROWS)
        # unread entries, -1 in sglang, gather slot 0 and weigh nothing
        rows = pool[torch.where(is_read[chunk], slots[chunk], 0)]
        codes = decode(rows[..., :_NOPE].view(torch.float8_e4m3fn), "float8_e4m3fn").to(q.dtype)
        scales = decode(rows[..., _NOPE:_SCALES].view(torch.float32), "float32").to(q.dtype)
        nope = (codes.unflatten(-1, (-1, _GROUP)) * scales[..., None]).flatten(-2)
        rope = decode(rows[..., _SCALES:].view(torch.bfloat16), "bfloat16").to(q.dtype)
        scores = torch.einsum("rhc,rkc->rhk", q[chunk], torch.cat((nope, rope), dim=-1)) * args["sm_scale"]
        scores = scores.masked_fill(~is_read[chunk, None], -torch.inf)
        lse[chunk] = scores.logsumexp(dim=-1)
        # a row that reads no key gets 0
        weights = scores.softmax(dim=-1).nan_to_num(0.0)
        output[chunk] = torch.einsum("rhk,rkc->rhc", weights, nope)
    return output, lse
