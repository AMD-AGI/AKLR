# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["biased_grouped_topk_reference"]


def biased_grouped_topk_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids): each token's topk experts by biased score.

    Every group is kept (the contract's constraint), so the groups mask nothing.
    """
    logits = to_float64(row_major_args["router_logits"], point.dtypes["router_logits"])
    bias = to_float64(row_major_args["correction_bias"], point.dtypes["correction_bias"])
    scores = logits.sigmoid()
    chosen = (scores + bias).topk(point.axes["topk"], dim=-1).indices
    weights = scores.gather(-1, chosen)
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * row_major_args["routed_scaling_factor"], chosen.double()
