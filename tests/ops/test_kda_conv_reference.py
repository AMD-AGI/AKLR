# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA short convolution reference against a plain causal conv1d (CPU)."""

import torch

from aklr.ops.kda_conv.reference import KimiK3KdaConvReference

PROJ, WIDTH, SLOTS = 128, 4, 4
DIM = 3 * PROJ


def test_the_reference_is_a_causal_conv1d_with_silu_from_the_state():
    """Against torch's conv1d over [state | inputs] per sequence; the slots end with each sequence's last inputs."""
    torch.manual_seed(0)
    n = 9
    x = torch.randn(n, 3, PROJ, dtype=torch.float32)
    weight = torch.randn(DIM, WIDTH)
    state = torch.randn(SLOTS, WIDTH - 1, DIM)
    args = dict(x=x, weight=weight, conv_state=state, state_indices=torch.tensor([1, 3]),
                has_initial_state=torch.tensor([True, False]), cu_seqlens=torch.tensor([0, 2, n]))  # fmt: skip
    out = KimiK3KdaConvReference().reference(**args)
    q, k, v, out_state = out["q"], out["k"], out["v"], out["conv_state"]
    flat = x.flatten(1)
    for start, end, slot, resume in [(0, 2, 1, True), (2, n, 3, False)]:
        head = state[slot] if resume else torch.zeros(WIDTH - 1, DIM)
        window = torch.cat((head, flat[start:end])).t()[None]  # (1, dim, len)
        z = torch.nn.functional.conv1d(window, weight[:, None, :], groups=DIM)[0].t()
        expected = torch.nn.functional.silu(z)
        got = torch.cat((q, k, v), dim=1)[start:end]
        torch.testing.assert_close(got, expected)
        assert torch.equal(
            out_state[slot], torch.cat((head, flat[start:end]))[-(WIDTH - 1) :].to(out_state.dtype)
        )
    assert torch.equal(out_state[0], state[0]) and torch.equal(out_state[2], state[2])
