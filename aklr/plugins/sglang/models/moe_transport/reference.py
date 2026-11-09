# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

__all__ = ["deepseek_v4_dp_moe_gather_reference"]

if TYPE_CHECKING:
    from torch import Tensor


def deepseek_v4_dp_moe_gather_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(out, hidden_states as the call leaves it): both hidden_states' rows with the rows at or
    past num_token_non_padded zero. No rounding: every value is copied or zero."""
    import torch

    hidden_states = decode(row_major_args["hidden_states"], point.dtypes["hidden_states"])
    n_real = int(row_major_args["num_token_non_padded"].reshape(-1)[0])
    padded = torch.arange(hidden_states.shape[0], device=hidden_states.device) >= n_real
    masked = hidden_states.masked_fill(padded[:, None], 0)
    return masked, masked.clone()
