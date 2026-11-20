# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aklr.plugins.aiter
from tests.op_sample import OpSample

_AXES = {"n_tokens": 9, "n_seqs": 2, "n_offsets": 3, "n_slots": 4, "proj": 128, "width": 4}
_SAMPLES = [
    OpSample(aklr.plugins.aiter.kimi_k3_kda_conv, "kimi_k3_kda_conv", _AXES),
    OpSample(aklr.plugins.aiter.kimi_k3_kda_conv, "kimi_k3_kda_conv_strided_rows", _AXES),
]


@pytest.mark.parametrize("sample", _SAMPLES, ids=lambda s: s.name)
def test_the_reference_is_a_causal_conv1d_with_silu_from_each_sequences_state(sample: OpSample) -> None:
    call = sample.call()
    assert sample.api.resolve({"q": None, "k": None, "v": None, **call}) is sample.template
    assert sample.schema().name == sample.name + "_proj128_width4"
    storage = call.pop("storage")
    q, k, v = sample.template.standard.reference(**call)
    rows, weight = call["x"].flatten(1).double(), call["weight"].double()
    offsets = call["cu_seqlens"].tolist()
    for seq, (slot, resumes) in enumerate(
        zip(call["state_indices"].tolist(), call["has_initial_state"].tolist(), strict=True)
    ):
        start, end = offsets[seq], offsets[seq + 1]
        head = call["conv_state"][slot].double()
        window = torch.cat((head if resumes else torch.zeros_like(head), rows[start:end])).t()[None]
        z = torch.nn.functional.conv1d(window, weight[:, None, :], groups=weight.shape[0])[0].t()
        torch.testing.assert_close(torch.cat((q, k, v), dim=1)[start:end], torch.nn.functional.silu(z))
    # vLLM's operands are rows of wider ones: dense, and more than a row apart
    assert call["x"].is_contiguous() == (storage == "row_major")
    assert call["conv_state"].is_contiguous() == (storage == "row_major")
    assert sample.template.standard.check([q, k, v], [o.to(torch.bfloat16) for o in (q, k, v)])
