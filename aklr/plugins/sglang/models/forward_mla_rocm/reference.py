# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in float32: the bf16 inputs decoded exactly, each dot
product summed in float32."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mla_v_absorb_reference"]


def mla_v_absorb_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """Per token and head, attn_output . w_vc^T; no rounding point."""
    import torch

    args = row_major_args
    latent = to_float32(args["attn_output"], point.dtypes["attn_output"])
    weight = to_float32(args["w_vc"], point.dtypes["w_vc"])
    return torch.einsum("thk,hvk->thv", latent, weight)
