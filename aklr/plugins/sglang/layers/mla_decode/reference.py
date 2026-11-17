# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What kimi_k3_mla_decode's contract says, in torch, in fp32."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_mla_decode_reference"]


def kimi_k3_mla_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each row's softmax over its slots of q . row * sm_scale, applied to the rows' first 512
    values; no rounding point."""
    import torch

    args = row_major_args
    q = decode(args["q"], point.dtypes["q"]).float()
    cache = decode(args["kv_cache"], point.dtypes["kv_cache"]).float()[:, 0]
    indptr, indices = args["kv_indptr"].long().tolist(), args["kv_indices"].long()
    output = q.new_zeros((q.shape[0], q.shape[1], 512))
    for r in range(q.shape[0]):
        rows = cache[indices[indptr[r] : indptr[r + 1]]]
        scores = (q[r] @ rows.T) * float(args["sm_scale"])
        output[r] = torch.softmax(scores, dim=-1) @ rows[:, :512]
    return output
