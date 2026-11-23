# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["fp8_mqa_logits_reference"]

# Per-head logits per chunk of rows: 1 GiB of float32
_CHUNK_ELEMENTS = 2**28


def fp8_mqa_logits_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """(no_values, logits): nothing, and logits with each row's window rewritten, the scale
    applied before relu as aiter does."""
    import torch

    q = to_float32(row_major_args["q"], point.dtypes["q"])
    kv = to_float32(row_major_args["kv"], point.dtypes["kv"])
    scales = to_float32(row_major_args["kv_scales"], "float32")
    weights = to_float32(row_major_args["weights"], "float32")
    starts = row_major_args["cu_starts"].long()
    ends = row_major_args["cu_ends"].long()
    out = to_float32(row_major_args["logits"], "float32").clone()
    n_q, n_heads = q.shape[:2]
    n_k = kv.shape[0]
    columns = torch.arange(n_k, device=q.device)
    n_rows = max(1, _CHUNK_ELEMENTS // (n_k * n_heads))
    for start in range(0, n_q, n_rows):
        rows = slice(start, start + n_rows)
        scores = torch.einsum("rhd,kd->rhk", q[rows], kv) * scales
        logits = torch.einsum("rhk,rh->rk", scores.relu(), weights[rows])
        window = (columns >= starts[rows, None]) & (columns < ends[rows, None])
        out[rows, :n_k] = torch.where(window, logits, out[rows, :n_k])
    return torch.zeros(0, dtype=out.dtype, device=out.device), out
