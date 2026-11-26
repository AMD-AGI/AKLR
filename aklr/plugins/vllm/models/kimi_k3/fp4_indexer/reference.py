# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The indexer prefill from its math, in float32: the queries' MXFP4 quantization (vLLM's rule), the keys
decoded from their pages, the scores summed head by head, and a stable sort for the top-k."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "HEAD_DIM",
    "KEYS_PER_PAGE",
    "MXFP4_GROUP",
    "SCALE_FLOOR",
    "KimiK3Fp4IndexerPrefillReference",
    "mxfp4_decoded",
]

# An index head and key: 128 values, four MXFP4 groups of 32
HEAD_DIM = 128
MXFP4_GROUP = 32
# Keys per staged page (aiter's 64-row pages)
KEYS_PER_PAGE = 64
# The query's scale floor before its power-of-two round-up (vLLM's _quantize)
SCALE_FLOOR = 1e-4

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# E2M1 codes 1..7 begin past these: the midpoints, a tie to the even code (> keeps it below, >= lifts it)
_CODE_STEPS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)
# float32 scores per step: rows of one step times the chunk's keys
_SCORES_PER_STEP = 2**25


def mxfp4_decoded(x: Tensor) -> Tensor:
    """``x`` (float32, last dim a multiple of 32) quantized per 32 values as vLLM's ``_quantize`` does, then decoded:
    a power-of-two scale 2^ceil(log2(max(amax / 6, 1e-4))) (its biased exponent at most 254), each value / scale
    rounded to the nearest E2M1 value, a tie to the even code; the result exact in float32."""
    import torch

    groups = x.unflatten(-1, (-1, MXFP4_GROUP))
    scale = torch.clamp(groups.abs().amax(dim=-1, keepdim=True) / 6.0, min=SCALE_FLOOR)
    bits = scale.view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(max=254)
    power = (exponent << 23).view(torch.float32)
    scaled = groups / power
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int64)
    for step, inclusive in _CODE_STEPS:
        code += (magnitude >= step) if inclusive else (magnitude > step)
    values = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)[code]
    return (torch.where(scaled < 0, -values, values) * power).flatten(-2)


def _keys(key_values: Tensor, key_scales: Tensor) -> Tensor:
    """Every staged key decoded, [n_pages * 64, 128] float32, key t at row t."""
    import torch

    values = to_float32(key_values, "float4_e2m1")  # [pages, 4, 64, 32]
    scales = to_float32(key_scales, "float8_e8m0")  # [pages, 4, 64], transposed 16 x 4
    slot = torch.arange(KEYS_PER_PAGE, device=scales.device)
    scales = scales[..., (slot % 16) * 4 + slot // 16]  # [pages, 4, 64] by slot
    keys = values * scales[..., None]
    return keys.permute(0, 2, 1, 3).reshape(-1, HEAD_DIM)


def _scores(
    q: Tensor,
    weights: Tensor,
    key_values: Tensor,
    key_scales: Tensor,
    row_starts: Tensor,
    row_ends: Tensor,
    weight_scale,
) -> Iterator[tuple[slice, Tensor, Tensor, Tensor]]:
    """(rows, scores, starts, ends) a few rows at a time: each row's float32 score of every key, -inf outside its
    keys [start, end); starts and ends as [rows, 1] int64."""
    import torch

    queries = mxfp4_decoded(q.float())  # [rows, heads, 128]
    head_weights = weights.float() * weight_scale
    keys = _keys(key_values, key_scales)
    starts, ends = row_starts.long()[:, None], row_ends.long()[:, None]
    n_rows, n_keys = queries.shape[0], keys.shape[0]
    columns = torch.arange(n_keys, device=keys.device)
    step = max(1, _SCORES_PER_STEP // n_keys)
    for first in range(0, n_rows, step):
        rows = slice(first, first + step)
        scores = torch.zeros((queries[rows].shape[0], n_keys), dtype=torch.float32, device=keys.device)
        for head in range(queries.shape[1]):
            scores += (queries[rows, head] @ keys.T).relu_() * head_weights[rows, head, None]
        start, end = starts[rows], ends[rows]
        yield rows, scores.masked_fill_((columns < start) | (columns >= end), -torch.inf), start, end


class KimiK3Fp4IndexerPrefillReference(OpReference):
    def reference(
        self, *, q, weights, key_values, key_scales, row_starts, row_ends, indices, weight_scale, **_
    ):
        """indices: each row's top-k keys by descending score, ties to the lower key, relative to its first key; a
        row of at most topk keys, all of them in order, then -1. For check: indices_scores, each row's scores by key
        relative to its first, -inf past the chunk's keys, rounded to float16; indices_full, the rows of more keys
        than topk."""
        import torch

        topk = indices.shape[1]
        n_rows, n_keys = q.shape[0], KEYS_PER_PAGE * key_values.shape[0]
        width = min(topk, n_keys)
        columns = torch.arange(width, device=q.device)
        out = torch.full((n_rows, topk), -1, dtype=torch.int64, device=q.device)
        relative = torch.empty((n_rows, n_keys), dtype=torch.float16, device=q.device)
        key_columns = torch.arange(n_keys, device=q.device)
        for rows, scores, start, end in _scores(
            q, weights, key_values, key_scales, row_starts, row_ends, weight_scale
        ):
            ranked = scores.argsort(dim=-1, descending=True, stable=True)[:, :width] - start
            row_lengths = (end - start).clamp(min=0)
            in_order = torch.where(columns < row_lengths, columns, -1)
            out[rows, :width] = torch.where(row_lengths <= topk, in_order, ranked)
            # key j of a row is key start + j; past the chunk's keys, -inf
            shifted = scores.gather(-1, (key_columns + start).clamp(max=n_keys - 1))
            relative[rows] = shifted.masked_fill_(key_columns + start >= n_keys, -torch.inf).to(torch.float16)
        lengths = row_ends.long() - row_starts.long()
        return {
            "indices": out.to(indices.dtype),
            "indices_scores": relative,
            "indices_full": (lengths > topk)[:, None],
        }

    def check(self, outputs, expected, schema):
        """``OpReference.check`` of each row's keys as a set, as vLLM takes them: both sides' rows put in one order
        (_in_one_order), then an id that differs from the reference's in a full row stands for it where the two keys'
        scores are equal or one float16 value apart, in a row without a repeated id. A kernel sums the heads in
        another order than the reference's, so a key tying the row's topk-th may take its place; float16 keeps them
        one value apart, while a score off by more than about 2^-11 of itself (a bfloat16 sum, a wrong scale) is not.
        A row of at most topk keys must hold every key, and -1 in its other slots."""
        import torch

        scores, full = expected["indices_scores"], expected["indices_full"]
        ids = _in_one_order(outputs["indices"], scores, full)
        reference_ids = _in_one_order(expected["indices"], scores, full)
        tied = full & _same_score(ids, reference_ids, scores)
        return super().check(
            {**outputs, "indices": torch.where(tied, reference_ids, ids)},
            {**expected, "indices": reference_ids},
            schema,
        )


def _in_one_order(ids: Tensor, scores: Tensor, full: Tensor) -> Tensor:
    """``ids`` with each row in one order: a full row by descending float16 score, then by key; any other row by key,
    its -1s (or any id past the keys) last."""
    import torch

    n = scores.shape[-1]
    a = ids.long()
    known = (a >= 0) & (a < n)
    by_key = torch.where(known, a, n).argsort(dim=-1, stable=True)
    a = a.gather(-1, by_key)
    known = known.gather(-1, by_key)
    score = scores.gather(-1, a.clamp(0, n - 1)).float().masked_fill_(~known, -torch.inf)
    by_score = a.gather(-1, score.argsort(dim=-1, descending=True, stable=True))
    return torch.where(full, by_score, a).to(ids.dtype)


def _same_score(ids: Tensor, reference_ids: Tensor, scores: Tensor) -> Tensor:
    """Where an id is in range, in a row without a repeated id, and its float16 score equals the reference id's or is
    the next float16 value above or below it."""
    import torch

    n = scores.shape[-1]
    a, y = ids.long(), reference_ids.long()  # gather's index dtype
    in_range = (a >= 0) & (a < n)
    a, y = a.clamp(0, n - 1), y.clamp(0, n - 1)
    # sign and magnitude to one integer line, +0 and -0 one value: adjacent values differ by 1
    bits = scores.view(torch.int16).long()
    order = torch.where(bits < 0, -(bits & 0x7FFF), bits & 0x7FFF)
    within = (order.gather(-1, a) - order.gather(-1, y)).abs() <= 1
    ordered = a.sort(-1).values
    distinct = (ordered[..., 1:] != ordered[..., :-1]).all(-1, keepdim=True)
    return in_range & distinct & within
