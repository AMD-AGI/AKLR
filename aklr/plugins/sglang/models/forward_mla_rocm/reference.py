# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in the dtype ``decode`` returns: the bf16 inputs decoded exactly,
each dot product summed in that dtype."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mla_v_absorb_reference"]


def mla_v_absorb_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Per token and head, attn_output . w_vc^T; no rounding point."""
    import torch

    args = row_major_args
    latent = decode(args["attn_output"], point.dtypes["attn_output"])
    weight = decode(args["w_vc"], point.dtypes["w_vc"])
    return torch.einsum("thk,hvk->thv", latent, weight)
