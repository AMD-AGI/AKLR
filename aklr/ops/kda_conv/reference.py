# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA's short convolution from its definition, sequence by sequence, in the dtype ``decode`` gives; the outputs' only
rounding is to x's dtype."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_conv_reference"]


def kimi_k3_kda_conv_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(q, k, v, conv_state) as the call leaves them."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    x = decode(args["x"], dtypes["x"]).double().flatten(1)  # (n_tokens, dim)
    weight = decode(args["weight"], dtypes["weight"]).double()  # (dim, width)
    conv_state = decode(args["conv_state"], dtypes["conv_state"]).clone()  # (n_slots, state_len, dim)
    state_len = conv_state.shape[1]
    offsets = args["cu_seqlens"].tolist()
    slots = args["state_indices"].tolist()
    resumes = args["has_initial_state"].tolist()
    y = torch.empty_like(x)
    for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
        start, end = offsets[seq], offsets[seq + 1]
        head = (
            conv_state[slot].double() if resume else torch.zeros_like(conv_state[slot], dtype=torch.float64)
        )
        window = torch.cat((head, x[start:end]))  # (state_len + length, dim)
        taps = torch.stack([window[j : j + end - start] for j in range(weight.shape[1])], dim=-1)
        z = (taps * weight).sum(dim=-1)
        y[start:end] = z * torch.sigmoid(z)
        conv_state[slot] = window[-state_len:].to(conv_state.dtype)
    proj = y.shape[1] // 3
    return y[:, :proj], y[:, proj : 2 * proj], y[:, 2 * proj :], conv_state
