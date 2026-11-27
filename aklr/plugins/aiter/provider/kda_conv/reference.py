# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA's short convolution from its definition, sequence by sequence, in float32; the outputs' only rounding is to
x's dtype."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["KimiK3KdaConvReference"]


class KimiK3KdaConvReference(OpReference):
    def reference(self, *, x, weight, conv_state, state_indices, has_initial_state, cu_seqlens, **_):
        """q, k, v and conv_state as the call leaves them."""
        dtype, x = x.dtype, x.float().flatten(1)  # (n_tokens, dim)
        weight = weight.float()  # (dim, width)
        conv_state = conv_state.float().clone()  # (n_slots, state_len, dim)
        state_len = conv_state.shape[1]
        offsets = cu_seqlens.tolist()
        slots = state_indices.tolist()
        resumes = has_initial_state.tolist()
        y = torch.empty_like(x)
        for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
            start, end = offsets[seq], offsets[seq + 1]
            head = conv_state[slot] if resume else torch.zeros_like(conv_state[slot])
            window = torch.cat((head, x[start:end]))  # (state_len + length, dim)
            taps = torch.stack([window[j : j + end - start] for j in range(weight.shape[1])], dim=-1)
            z = (taps * weight).sum(dim=-1)
            y[start:end] = z * torch.sigmoid(z)
            conv_state[slot] = window[-state_len:]
        y, conv_state = y.to(dtype), conv_state.to(dtype)
        proj = y.shape[1] // 3
        return {"q": y[:, :proj], "k": y[:, proj : 2 * proj], "v": y[:, 2 * proj :], "conv_state": conv_state}
