# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The aggregation from its math, in float64; the only rounding is p's, to its dtype."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_attn_res_reference", "attn_res_reference"]

_CHUNK_TOKENS = 2048  # float64 rows of 9 x 8192 per token: about 1.2 GB a chunk


def attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(normalized, bank) over the prefix as it is."""
    args, dtypes = row_major_args, point.dtypes
    return _aggregate(point, args, decode, decode(args["prefix"], dtypes["prefix"]).double())


def add_attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(normalized, prefix_out, bank) over prefix + addend rounded to the prefix's dtype."""
    args, dtypes = row_major_args, point.dtypes
    total = (
        decode(args["prefix"], dtypes["prefix"]).double() + decode(args["addend"], dtypes["addend"]).double()
    )
    prefix = total.to(args["prefix"].dtype).double()
    normalized, bank = _aggregate(point, args, decode, prefix)
    return normalized, prefix, bank


def _aggregate(
    point: Any, args: Mapping[str, Any], decode: Callable, prefix: Tensor
) -> tuple[Tensor, Tensor]:
    import torch

    dtypes, valid = point.dtypes, point.attrs["n_valid_rows"]
    bank = decode(args["bank"], dtypes["bank"]).double()
    score_weight = decode(args["score_weight"], dtypes["score_weight"]).double()
    norm_weight = decode(args["norm_weight"], dtypes["norm_weight"]).double()
    normalized = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], _CHUNK_TOKENS):
        tokens = slice(start, start + _CHUNK_TOKENS)
        rows = torch.cat((bank[tokens, :valid], prefix[tokens, None]), dim=1)
        scores = (rows @ score_weight) * (rows.square().mean(dim=-1) + args["score_eps"]).rsqrt()
        mix = (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)
        normalized[tokens] = (
            mix * (mix.square().mean(dim=-1, keepdim=True) + args["norm_eps"]).rsqrt() * norm_weight
        )
    if point.attrs["write_bank"]:
        bank[:, valid] = prefix
    return normalized, bank
