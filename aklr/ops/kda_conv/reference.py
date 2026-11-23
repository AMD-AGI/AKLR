# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA's short convolution from its definition, sequence by sequence, in float32; the outputs' only rounding is to
x's dtype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_conv_reference"]


def kimi_k3_kda_conv_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(q, k, v, conv_state) as the call leaves them."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    x = to_float32(args["x"], dtypes["x"]).flatten(1)  # (n_tokens, dim)
    weight = to_float32(args["weight"], dtypes["weight"])  # (dim, width)
    conv_state = to_float32(args["conv_state"], dtypes["conv_state"]).clone()  # (n_slots, state_len, dim)
    state_len = conv_state.shape[1]
    offsets = args["cu_seqlens"].tolist()
    slots = args["state_indices"].tolist()
    resumes = args["has_initial_state"].tolist()
    y = torch.empty_like(x)
    for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
        start, end = offsets[seq], offsets[seq + 1]
        head = conv_state[slot] if resume else torch.zeros_like(conv_state[slot])
        window = torch.cat((head, x[start:end]))  # (state_len + length, dim)
        taps = torch.stack([window[j : j + end - start] for j in range(weight.shape[1])], dim=-1)
        z = (taps * weight).sum(dim=-1)
        y[start:end] = z * torch.sigmoid(z)
        conv_state[slot] = window[-state_len:]
    proj = y.shape[1] // 3
    return y[:, :proj], y[:, proj : 2 * proj], y[:, 2 * proj :], conv_state
