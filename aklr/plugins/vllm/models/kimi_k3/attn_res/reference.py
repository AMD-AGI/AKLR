# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The attention residual from its math, in float64; the only rounding is p's, to its dtype."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_attn_res_reference"]

_CHUNK_TOKENS = 2048  # float64 rows of 9 x 8192 per token: about 1.2 GB a chunk


def kimi_k3_attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(out, prefix, bank) as the call leaves them."""
    import torch

    args, dtypes, attrs = row_major_args, point.dtypes, point.attrs
    valid = attrs["n_valid_rows"]
    prefix = decode(args["prefix"], dtypes["prefix"]).double()
    if attrs["has_addend"]:
        total = prefix + decode(args["addend"], dtypes["addend"]).double()
        prefix = total.to(args["prefix"].dtype).double()
    bank = decode(args["bank"], dtypes["bank"]).double()
    score_weight = (
        decode(args["score_norm_weight"], dtypes["score_norm_weight"]).double()
        * decode(args["score_proj_weight"], dtypes["score_proj_weight"]).double()
    )
    norm_weight = decode(args["norm_weight"], dtypes["norm_weight"]).double() if attrs["has_norm"] else None
    out = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], _CHUNK_TOKENS):
        tokens = slice(start, start + _CHUNK_TOKENS)
        rows = torch.cat((bank[tokens, :valid], prefix[tokens, None]), dim=1)
        scores = (rows @ score_weight) * (rows.square().mean(dim=-1) + args["score_eps"]).rsqrt()
        mix = (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)
        if norm_weight is not None:
            mix = mix * (mix.square().mean(dim=-1, keepdim=True) + args["norm_eps"]).rsqrt() * norm_weight
        out[tokens] = mix
    if attrs["write_bank"]:
        bank[:, valid] = prefix
    return out, prefix, bank
