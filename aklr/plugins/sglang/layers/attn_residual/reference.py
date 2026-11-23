# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The aggregation from its math, in float32; the roundings are p's and the outputs', to the prefix's dtype."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["AddAttnResReference", "AttnResReference"]

_CHUNK_TOKENS = 2048  # float32 rows of 9 x 8192 per token: about 0.6 GB a chunk


class AttnResReference(OpReference):
    def reference(
        self, *, prefix, bank, score_weight, norm_weight, score_eps, norm_eps, n_valid_rows, write_bank, **_
    ):
        """normalized, then bank, over the prefix as it is."""
        normalized, bank = _aggregate(
            prefix.float(), bank, score_weight, norm_weight, score_eps, norm_eps, n_valid_rows, write_bank
        )
        return {"normalized": normalized.to(prefix.dtype), "bank": bank.to(prefix.dtype)}


class AddAttnResReference(OpReference):
    def reference(
        self,
        *,
        prefix,
        addend,
        bank,
        score_weight,
        norm_weight,
        score_eps,
        norm_eps,
        n_valid_rows,
        write_bank,
        **_,
    ):
        """normalized, prefix_out, then bank, over prefix + addend rounded to the prefix's dtype."""
        prefix_out = (prefix.float() + addend.float()).to(prefix.dtype)
        normalized, bank = _aggregate(
            prefix_out.float(), bank, score_weight, norm_weight, score_eps, norm_eps, n_valid_rows, write_bank
        )
        return {
            "normalized": normalized.to(prefix.dtype),
            "prefix_out": prefix_out,
            "bank": bank.to(prefix.dtype),
        }


def _aggregate(prefix, bank, score_weight, norm_weight, score_eps, norm_eps, valid, write_bank):
    bank, score_weight, norm_weight = bank.float(), score_weight.float(), norm_weight.float()
    normalized = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], _CHUNK_TOKENS):
        tokens = slice(start, start + _CHUNK_TOKENS)
        rows = torch.cat((bank[tokens, :valid], prefix[tokens, None]), dim=1)
        scores = (rows @ score_weight) * (rows.square().mean(dim=-1) + score_eps).rsqrt()
        mix = (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)
        normalized[tokens] = mix * (mix.square().mean(dim=-1, keepdim=True) + norm_eps).rsqrt() * norm_weight
    if write_bank:
        bank[:, valid] = prefix
    return normalized, bank
