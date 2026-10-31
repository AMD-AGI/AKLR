# Created by OpenAI
import pytest
import torch

from aklr.plugins.sglang.models.qwen3_5.reference import reference_prefill


def arguments(rows=3, lengths=(3,)):
    torch.manual_seed(7)
    # A small mathematical example, independent of the production specialization.
    key_heads, heads, dim = 1, 2, 4
    channels = (2 * key_heads + heads) * dim
    return {
        "projected_qkvz": (torch.randn(rows, channels + heads * dim) * 0.2).bfloat16(),
        "projected_ba": (torch.randn(rows, heads * 2) * 0.2).bfloat16(),
        "conv_state": (torch.randn(3, channels, 3) * 0.2).bfloat16(),
        "delta_state": torch.randn(3, heads, dim, dim) * 0.1,
        "cache_indices": torch.tensor([1] if len(lengths) == 1 else [1, 2], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32),
        "has_initial_state": torch.tensor([True] * len(lengths)),
        "conv_weight": (torch.randn(channels, 4) * 0.2).bfloat16(),
        "conv_bias": torch.zeros(channels, dtype=torch.bfloat16),
        "a_log": torch.full((heads,), -2.0),
        "dt_bias": torch.zeros(heads, dtype=torch.bfloat16),
        "norm_weight": torch.ones(dim, dtype=torch.bfloat16),
        "scale": dim**-0.5,
    }


def test_reference_does_not_modify_arguments_or_unselected_slots():
    args = arguments()
    snapshots = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    out, conv, state, quant, scales = reference_prefill(**args)
    for key, before in snapshots.items():
        assert torch.equal(args[key], before)
    assert torch.equal(conv[[0, 2]], args["conv_state"][[0, 2]])
    assert torch.equal(state[[0, 2]], args["delta_state"][[0, 2]])
    assert out.shape == (3, 2, 4) and quant.shape == (3, 8)
    assert scales.shape == (3, 2) and scales.gt(0).all()


def test_fresh_sequence_ignores_dirty_cached_state():
    args = arguments()
    args["has_initial_state"][:] = False
    first = reference_prefill(**args)
    args["conv_state"].fill_(100)
    args["delta_state"].fill_(-100)
    second = reference_prefill(**args)
    for i in (0, 3, 4):
        actual, expected = first[i], second[i]
        if actual.dtype == torch.float8_e4m3fn:
            actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
        assert torch.equal(actual, expected)
    assert torch.equal(first[1][1], second[1][1])
    assert torch.equal(first[2][1], second[2][1])


def test_zero_value_produces_zero_output_and_finite_positive_scales():
    args = arguments()
    args["projected_qkvz"].zero_()
    args["has_initial_state"][:] = False
    out, _, state, quant, scales = reference_prefill(**args)
    assert out.eq(0).all() and state[1].eq(0).all()
    assert quant.float().eq(0).all()
    assert torch.isfinite(scales).all() and scales.gt(0).all()


@pytest.mark.parametrize("bad", ["duplicate", "negative", "out_of_range", "starts"])
def test_reference_rejects_bad_state_slot_or_sequence_metadata(bad):
    args = arguments(rows=3, lengths=(1, 2))
    if bad == "duplicate":
        args["cache_indices"][:] = 1
    elif bad == "negative":
        args["cache_indices"][0] = -1
    elif bad == "out_of_range":
        args["cache_indices"][0] = 3
    else:
        args["cu_seqlens"][-1] = 2
    with pytest.raises(ValueError):
        reference_prefill(**args)
