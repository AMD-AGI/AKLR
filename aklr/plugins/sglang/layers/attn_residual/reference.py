# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The aggregation from its math, in float32; the only rounding is p's, to its dtype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["add_attn_res_reference", "attn_res_reference"]

_CHUNK_TOKENS = 2048  # float32 rows of 9 x 8192 per token: about 0.6 GB a chunk


def attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """(normalized, bank) over the prefix as it is."""
    args, dtypes = row_major_args, point.dtypes
    return _aggregate(point, args, to_float32(args["prefix"], dtypes["prefix"]))


def add_attn_res_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """(normalized, prefix_out, bank) over prefix + addend rounded to the prefix's dtype."""
    args, dtypes = row_major_args, point.dtypes
    total = to_float32(args["prefix"], dtypes["prefix"]) + to_float32(args["addend"], dtypes["addend"])
    prefix = total.to(args["prefix"].dtype).float()
    normalized, bank = _aggregate(point, args, prefix)
    return normalized, prefix, bank


def _aggregate(point: Any, args: Mapping[str, Any], prefix: Tensor) -> tuple[Tensor, Tensor]:
    import torch

    dtypes, valid = point.dtypes, point.attrs["n_valid_rows"]
    bank = to_float32(args["bank"], dtypes["bank"])
    score_weight = to_float32(args["score_weight"], dtypes["score_weight"])
    norm_weight = to_float32(args["norm_weight"], dtypes["norm_weight"])
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
