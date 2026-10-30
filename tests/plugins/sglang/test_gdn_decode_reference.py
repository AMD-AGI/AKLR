# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.plugins.sglang.models.qwen3_5.reference import (
    reference_decode,
    reference_prefill,
)


def arguments(indices=(1, -1, 3)):
    torch.manual_seed(11)
    # A small mathematical example, independent of the production specialization.
    key_heads, heads, dim, slots = 1, 2, 4, 5
    channels = (2 * key_heads + heads) * dim
    rows = len(indices)
    return {
        "projected_qkvz": (torch.randn(rows, channels + heads * dim) * 0.5).bfloat16(),
        "projected_ba": (torch.randn(rows, heads * 2) * 0.5).bfloat16(),
        "conv_state": (torch.randn(slots, channels, 3) * 0.5).bfloat16(),
        "delta_state": torch.randn(slots, heads, dim, dim) * 0.1,
        "cache_indices": torch.tensor(indices, dtype=torch.int32),
        "conv_weight": (torch.randn(channels, 4) * 0.5).bfloat16(),
        "a_log": torch.full((heads,), -1.0),
        "dt_bias": torch.zeros(heads, dtype=torch.bfloat16),
        "norm_weight": (1 + torch.randn(dim) * 0.1).bfloat16(),
        "scale": dim**-0.5,
    }


def test_padding_rows_and_unindexed_slots_keep_their_state():
    args = arguments((1, -1, 3, -1))
    snapshots = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    normalized, conv, state, values, scales = reference_decode(**args)
    for key, before in snapshots.items():
        assert torch.equal(args[key], before), key
    for slot in (0, 2, 4):
        assert torch.equal(conv[slot], args["conv_state"][slot])
        assert torch.equal(state[slot], args["delta_state"][slot])
    for slot in (1, 3):
        assert not torch.equal(state[slot], args["delta_state"][slot])
        # the history shifts the row's Q/K/V channels in
        assert torch.equal(conv[slot, :, :2], args["conv_state"][slot, :, 1:])
    assert normalized.shape == (4, 2, 4) and values.shape == (4, 8)
    assert scales.shape == (4, 2)
    for row in (1, 3):
        assert normalized[row].float().eq(0).all() and values[row].float().eq(0).all()
        assert torch.equal(scales[row], torch.full((2,), 1e-10 / 448, dtype=scales.dtype))
    assert scales[[0, 2]].gt(1e-10 / 448).all()


def test_grouped_values_dequantize_to_the_bf16_output():
    normalized, _, _, values, scales = reference_decode(**arguments())
    dequantized = values.double().reshape(3, 2, 4) * scales[..., None]
    torch.testing.assert_close(dequantized, normalized.double(), atol=0, rtol=0.07)
    expected = normalized.double().abs().amax(-1).clamp_min(1e-10) / 448
    assert torch.equal(scales, expected)


def test_decode_is_prefill_of_one_token_sequences():
    """Both references describe one layer: prefill also rounds Q and K to BF16, and
    decode each convolution tap's product, within decode's bounds."""
    args = arguments((3, 0, 4))
    decoded = reference_decode(**args)
    prefill = reference_prefill(
        **{k: v for k, v in args.items() if k != "cache_indices"},
        cache_indices=args["cache_indices"],
        cu_seqlens=torch.arange(4, dtype=torch.int32),
        has_initial_state=torch.ones(3, dtype=torch.bool),
        conv_bias=torch.zeros(args["conv_weight"].shape[0], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(decoded[0].float(), prefill[0].float(), atol=0.005, rtol=0.02)
    assert torch.equal(decoded[1], prefill[1])
    torch.testing.assert_close(decoded[2], prefill[2], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bad", ["duplicate", "below_padding", "out_of_range", "rows"])
def test_reference_rejects_bad_state_slots(bad):
    args = arguments()
    if bad == "duplicate":
        args["cache_indices"][2] = 1
    elif bad == "below_padding":
        args["cache_indices"][0] = -2
    elif bad == "out_of_range":
        args["cache_indices"][0] = 5
    else:
        args["cache_indices"] = args["cache_indices"][:2]
    with pytest.raises(ValueError):
        reference_decode(**args)
