# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.plugins.sglang.models.deepseek_v2.baseline import deepseek_v2_fp8_moe_baseline
from tests.op_case import OpCase
from tests.plugins.sglang.cases import SMALL_POINTS


@pytest.mark.gfx950
def test_fp8_baseline_preserves_fp32_expert_selection():
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "deepseek_v2_fp8_moe@1.0" and p.attrs["num_expert_group"] == 1
    )
    point = point.model_copy(
        update={
            "axes": point.axes | {"n_tokens": 1},
            "dtypes": point.dtypes | {"correction_bias": "float32"},
        }
    )
    case = OpCase(point)
    inputs = case.inputs()
    inputs["hidden_states"].fill_(1)
    inputs["router_weight"].zero_()
    inputs["router_weight"][-1, 0] = 1 / 256
    inputs["correction_bias"].fill_(0.125)
    inputs["correction_bias"][1:8] = 2
    inputs["correction_bias"][0] = 1.002
    inputs["correction_bias"][-1] = 1
    inputs["w1"].zero_()
    inputs["w2"].zero_()
    inputs["w1"][0].fill_(1)
    inputs["w2"][0].fill_(1)
    inputs["w1_scale"].fill_(1)
    inputs["w2_scale"].fill_(1)

    actual = deepseek_v2_fp8_moe_baseline(**case.call(inputs, "cuda"))

    # Seven certain experts 1..7 plus expert0: its +.002 FP32 bias beats the
    # last expert's sigmoid(1/256). BF16 rounds that bias to1 and loses expert0.
    # Only expert0 has nonzero weights, so native FP32 routing gives positive output.
    assert torch.all(actual > 1)
