# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.cost import Cost
from tests.op_case import OpCase


class TestCost:
    """``OpContract.cost_at``: the op's ``cost.py`` function at a Point."""

    def test_gemm_does_2mnk_and_moves_each_tensor_once(self) -> None:
        """m 4, n 16, k 128 in bf16, no bias: a, b and out move once."""
        case = OpCase.of("gemm@1.0")
        assert case.contract.cost_at(case.point) == Cost(
            2 * 4 * 16 * 128, (4 * 128 + 16 * 128 + 4 * 16) * 2, "bfloat16"
        )

    def test_fused_moe_moves_only_reached_experts(self) -> None:
        """4 tokens, top-2 of 4 experts: all but (1 - 2/4)**4 of the expert weights move;
        fp4 weights and their e8m0 scales, 4-bit activations at the fp4 rate."""
        case = OpCase.of("fused_moe@1.0")
        cost = case.contract.cost_at(case.point)
        assert cost is not None
        weights = 4 * 512 * 256 // 2 + 4 * 256 * 256 // 2  # w1 and w2, two fp4 values per byte
        scales = 4 * 512 * 8 + 4 * 256 * 8  # one byte per 32 values
        rest = 4 * 256 * 2 + 4 * 2 * 4 + 4 * 2 * 4 + 4 * 256 * 2  # hidden_states, topk weights and ids, out
        assert cost.bytes == rest + round((1 - 0.5**4) * (weights + scales))
        assert cost.flops == 2 * 4 * 2 * 256 * (512 + 256)
        assert cost.dtype == "float4_e2m1"

    def test_norm_is_bytes_alone(self) -> None:
        case = OpCase.of("rms_norm@1.0")
        assert case.contract.cost_at(case.point) == Cost(0, (2 * 4 * 2048 + 2048) * 2, None)

    def test_an_op_without_a_cost_function_has_none(self) -> None:
        case = OpCase.of("unified_attention@1.0")
        assert case.contract.cost_at(case.point) is None
