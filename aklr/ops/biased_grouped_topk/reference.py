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
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids): each token's topk experts by biased score, ties to the lower expert.

    Every group is kept (the contract's constraint), so the groups mask nothing.
    """
    # scored in float64: float32 ties distinct scores, and the ids compare exactly
    logits = decode(row_major_args["router_logits"], point.dtypes["router_logits"]).double()
    bias = decode(row_major_args["correction_bias"], point.dtypes["correction_bias"]).double()
    scores = logits.sigmoid()
    # torch.topk orders equal values as its algorithm happens to: a stable sort fixes them
    ranked = (scores + bias).sort(dim=-1, descending=True, stable=True).indices
    chosen = ranked[..., : point.axes["topk"]]
    weights = scores.gather(-1, chosen)
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * row_major_args["routed_scaling_factor"], chosen.double()
