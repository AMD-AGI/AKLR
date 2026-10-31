# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["eagle_worker_v2_draft_top1_reference"]


def eagle_worker_v2_draft_top1_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(1, torch.argmax of each row); fp32 holds the logits exactly, their order and
    ties."""
    import torch

    logits = to_float64(row_major_args["logits"], point.dtypes["logits"]).float()
    index = logits.argmax(dim=-1, keepdim=True).double()
    return torch.ones_like(index), index
