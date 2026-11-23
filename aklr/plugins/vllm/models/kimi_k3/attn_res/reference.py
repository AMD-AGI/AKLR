# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The attention residual from its math, in float32; the only rounding is p's, to its dtype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_attn_res_reference"]

_CHUNK_TOKENS = 2048  # float32 rows of 9 x 8192 per token: about 0.6 GB a chunk


def kimi_k3_attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """(out, prefix, bank) as the call leaves them."""
    import torch

    args, dtypes, attrs = row_major_args, point.dtypes, point.attrs
    valid = attrs["n_valid_rows"]
    prefix = to_float32(args["prefix"], dtypes["prefix"])
    if attrs["has_addend"]:
        total = prefix + to_float32(args["addend"], dtypes["addend"])
        prefix = total.to(args["prefix"].dtype).float()
    bank = to_float32(args["bank"], dtypes["bank"])
    score_norm = to_float32(args["score_norm_weight"], dtypes["score_norm_weight"])
    score_weight = score_norm * to_float32(args["score_proj_weight"], dtypes["score_proj_weight"])
    norm_weight = to_float32(args["norm_weight"], dtypes["norm_weight"]) if attrs["has_norm"] else None
    out = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], _CHUNK_TOKENS):
        tokens = slice(start, start + _CHUNK_TOKENS)
        if valid == 0:
            # one source, whose softmax weight is 1 whatever its score (an infinite p scores NaN)
            mix = prefix[tokens]
        else:
            rows = torch.cat((bank[tokens, :valid], prefix[tokens, None]), dim=1)
            scores = (rows @ score_weight) * (rows.square().mean(dim=-1) + args["score_eps"]).rsqrt()
            mix = (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)
        if norm_weight is not None:
            mix = mix * (mix.square().mean(dim=-1, keepdim=True) + args["norm_eps"]).rsqrt() * norm_weight
        out[tokens] = mix
    if attrs["write_bank"]:
        bank[:, valid] = prefix
    return out, prefix, bank
