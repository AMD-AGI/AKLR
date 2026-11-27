# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.plugins.sglang.models.deepseek_v2.contract import DEEPSEEK_V2_FP8_MOE
from tests.op_case import OpCase
from tests.plugins.sglang.cases import SMALL_POINTS


@pytest.mark.parametrize("bias_dtype", ["bfloat16", "float32"])
def test_fp8_moe_accepts_native_correction_bias_dtype(bias_dtype):
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "deepseek_v2_fp8_moe@1.0" and p.attrs["num_expert_group"] == 1
    )
    point = point.model_copy(update={"dtypes": point.dtypes | {"correction_bias": bias_dtype}})
    assert DEEPSEEK_V2_FP8_MOE.domain.contains(point)


@pytest.mark.parametrize(
    ("bias_dtype", "median_lo", "median_hi"),
    [("bfloat16", -1, 1), ("float32", 4, 16)],
)
def test_fp8_bias_samples_cover_the_native_checkpoint_scale(bias_dtype, median_lo, median_hi):
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "deepseek_v2_fp8_moe@1.0" and p.attrs["num_expert_group"] == 1
    )
    point = point.model_copy(update={"dtypes": point.dtypes | {"correction_bias": bias_dtype}})
    bias = OpCase(point).inputs()["correction_bias"]
    # The new FP32 GLM input is positive around8 within a layer; the legacy
    # BF16 calibration remains near0. Neither is a zeroed routing bias.
    assert median_lo < bias.float().median().item() < median_hi
