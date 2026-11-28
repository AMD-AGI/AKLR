# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from tests.op_case import OpCase

GATED_ATTENTION = [
    "gated_attention_decode@1.0",
    "gated_attention_decode_group_fp8_quant@1.0",
    "gated_attention_prefill@1.0",
    "gated_attention_prefill_group_fp8_quant@1.0",
]


@pytest.mark.parametrize("contract_id", GATED_ATTENTION)
def test_positions_stay_two_rows_inside_the_cos_sin_table(contract_id: str) -> None:
    """sglang serves positions up to the table's rows - 3 (a request holds at most context_len - 2
    tokens), and its fused mRoPE reads 160 floats past a temporal position's row: every drawn
    position leaves those two rows inside the table."""
    case = OpCase.of(contract_id)
    for seed in range(8):
        args = case.inputs(f"numeric/{seed}")
        assert int(args["positions"].max()) <= args["cos_sin_cache"].shape[0] - 3


@pytest.mark.parametrize("contract_id", GATED_ATTENTION)
def test_key_cache_admits_one_bf16_ulp_before_rope(contract_id: str) -> None:
    """A key element of RoPE's 2-4 magnitude may round one BF16 ulp (2**-6) away from the
    reference's before its rotation, and moves a cancelling rotated element near zero by as much:
    that key passes. A key rotated at the next position still fails."""
    case = OpCase.of(contract_id)
    args = case.inputs()
    reference = case.reference_outputs(args)
    slots = args["cache_locations"]
    written = reference["key_cache"][slots].float()
    row, head, channel = (written.abs() == written.abs().min()).nonzero()[0].tolist()
    assert abs(written[row, head, channel]) < 2**-6
    nudged = reference["key_cache"].clone()
    nudged[slots[row], head, channel] = (written[row, head, channel] + 2**-6).to(nudged.dtype)
    assert "key_cache" not in case.compare({**reference, "key_cache": nudged}, args)

    moved = {**args, "positions": args["positions"] + 1}
    assert "key_cache" in case.compare(
        {**reference, "key_cache": case.reference_outputs(moved)["key_cache"]}, args
    )


@pytest.mark.parametrize(
    "contract_id",
    ["gated_attention_decode_group_fp8_quant@1.0", "gated_attention_prefill_group_fp8_quant@1.0"],
)
def test_scales_admit_the_e4m3_query_of_long_rows(contract_id: str) -> None:
    """On rows past 16384 tokens the kernels round Q to E4M3, as sglang does, and their scales
    come to 13.4 dB at worst (sglang's own: 13.4 dB, 1042 draws at 131072 tokens): scales 21%
    off each way (13.6 dB) pass. Ignoring the gate still fails."""
    case = OpCase.of(contract_id)
    args = case.inputs()
    reference = case.reference_outputs(args)
    scales = reference["scales"]
    sign = torch.ones_like(scales).flatten()
    sign[1::2] = -1
    off = scales * (1 + 0.21 * sign.reshape(scales.shape))
    assert "scales" not in case.compare({**reference, "scales": off}, args)

    projected = args["projected_qkv"].clone()
    head_dim = args["q_norm_weight"].shape[0]
    n_heads = reference["values"].shape[1]
    for head in range(n_heads):
        projected[:, (2 * head + 1) * head_dim : (2 * head + 2) * head_dim] = 30
    ungated = case.reference_outputs({**args, "projected_qkv": projected})
    assert case.compare(ungated, args)
