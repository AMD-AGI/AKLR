###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_conv_reference"]


def kimi_k3_kda_conv_reference(
    x, weight, conv_state, state_indices, has_initial_state, cu_seqlens
) -> tuple[Tensor, Tensor, Tensor]:
    """(q, k, v) from the convolution's definition, sequence by sequence in float64."""
    import torch

    rows = x.double().flatten(1)  # (n_tokens, dim)
    taps = weight.double()  # (dim, width)
    offsets = cu_seqlens.tolist()
    y = torch.empty_like(rows)
    for seq, (slot, resumes) in enumerate(
        zip(state_indices.tolist(), has_initial_state.tolist(), strict=True)
    ):
        start, end = offsets[seq], offsets[seq + 1]
        head = conv_state[slot].double()
        window = torch.cat((head if resumes else torch.zeros_like(head), rows[start:end]))
        z = sum(window[j : j + end - start] * taps[:, j] for j in range(taps.shape[1]))
        y[start:end] = z * torch.sigmoid(z)
    proj = y.shape[1] // 3
    return y[:, :proj], y[:, proj : 2 * proj], y[:, 2 * proj :]
