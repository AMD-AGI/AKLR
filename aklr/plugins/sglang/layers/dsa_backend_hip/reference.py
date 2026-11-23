# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The sparse MLA prefill from its math, in float32: the fp8_ds_mla rows decoded exactly (fp8 codes times
their group's scale, the bf16 RoPE values), nothing rounded in between; independent of the kernels and of
sglang's."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["dsa_sparse_mla_fp8_ds_prefill_reference"]

# tokens a step: a chunk gathers [tokens, topk, 656] bytes and decodes them to [tokens, topk, 576]
_CHUNK_TOKENS = 16
_NOPE, _GROUP, _ROPE_START = 512, 128, 528


def dsa_sparse_mla_fp8_ds_prefill_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """Each head's softmax over the scores q . key * sm_scale of the keys its token's indices name (-1
    none), weighing the keys' NoPE values, and the natural log of the sum of exp(score); 0 and -inf for a
    token that reads no key."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    q = torch.cat(
        (to_float32(args["q_nope"], dtypes["q_nope"]), to_float32(args["q_rope"], dtypes["q_rope"])), -1
    )
    pool = args["kv_cache"].view(torch.uint8)
    slots = args["indices"].long()
    is_read = slots >= 0
    output = q.new_zeros((*q.shape[:2], _NOPE))
    lse = q.new_empty(q.shape[:2])
    for start in range(0, q.shape[0], _CHUNK_TOKENS):
        chunk = slice(start, start + _CHUNK_TOKENS)
        # an unread entry gathers slot 0 and weighs nothing
        rows = pool[torch.where(is_read[chunk], slots[chunk], 0)]
        codes = to_float32(rows[..., :_NOPE].contiguous().view(torch.float8_e4m3fn), "float8_e4m3fn")
        scales = rows[..., _NOPE:_ROPE_START].contiguous().view(torch.float32)
        nope = (codes.unflatten(-1, (-1, _GROUP)) * scales[..., None]).flatten(-2)
        rope = to_float32(rows[..., _ROPE_START:].contiguous().view(torch.bfloat16), "bfloat16")
        scores = torch.einsum("thc,tkc->thk", q[chunk], torch.cat((nope, rope), -1)) * float(args["sm_scale"])
        scores = scores.masked_fill(~is_read[chunk, None], -torch.inf)
        lse[chunk] = scores.logsumexp(dim=-1)
        # a token that reads no key gets 0
        weights = scores.softmax(dim=-1).nan_to_num(0.0)
        output[chunk] = torch.einsum("thk,tkc->thc", weights, nope)
    return output, lse
