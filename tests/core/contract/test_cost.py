# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.contract.cost import Cost
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import contract_name
from aklr.ops.cost import COSTS, MEMORY_BOUND, fused_moe_cost
from tests.op_case import OpCase


class TestCost:
    """``OpContract.cost_at``: the contract's ``aklr/ops/cost.py`` formula at a Point, or its tensors moved once."""

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
        weights = 4 * 512 * 256 // 2 + 4 * 256 * 256 // 2  # w1 and w2, two fp4 values per byte
        scales = 4 * 512 * 8 + 4 * 256 * 8  # one byte per 32 values
        rest = 4 * 256 * 2 + 4 * 2 * 4 + 4 * 2 * 4 + 4 * 256 * 2  # hidden_states, topk weights and ids, out
        assert cost.bytes_moved == rest + round((1 - 0.5**4) * (weights + scales))
        assert cost.flops == 2 * 4 * 2 * 256 * (512 + 256)
        assert cost.flops_dtype == "float4_e2m1"

    @pytest.mark.parametrize(
        "gate_mode, activation, act_quant, n_tokens, dtype",
        [
            ("separated", "silu", 4, 8192, "float4_e2m1"),
            ("separated", "situv2", 8, 8192, "float8_e4m3fn"),
            ("separated", "swiglu", 4, 255, "bfloat16"),
            ("separated", "swiglu", 4, 256, "float4_e2m1"),
            # aiter's interleave goes fp8 past its row bound, and the bound admits fp8 at any count
            ("interleave", "silu", 4, 1, "float8_e4m3fn"),
            ("interleave", "silu", 4, 16384, "float8_e4m3fn"),
        ],
    )
    def test_fused_moe_math_runs_at_the_least_precision_its_bound_admits(
        self, gate_mode: str, activation: str, act_quant: int, n_tokens: int, dtype: str
    ) -> None:
        symbols = {
            "n_tokens": n_tokens, "topk": 1, "n_experts": 1, "model_dim": 1, "inter_dim": 1,
            "gate_up_rows": 2, "global_experts": 1, "has_expert_mask": False,
            "gate_mode": gate_mode, "activation": activation, "act_quant": act_quant,
        }  # fmt: skip
        assert fused_moe_cost(symbols, {"w1": 0, "w2": 0, "w1_scale": 0, "w2_scale": 0}).flops_dtype == dtype

    def test_norm_is_bytes_alone(self) -> None:
        case = OpCase.of("rms_norm@1.0")
        assert case.contract.cost_at(case.point) == Cost(0, (2 * 4 * 2048 + 2048) * 2, None)

    def test_every_name_is_a_contracts(self) -> None:
        """``cost_at`` reads ``COSTS`` and ``MEMORY_BOUND`` by contract name: a name of no
        contract is a misspelling that leaves the contract with the default cost."""
        names = {contract_name(c.contract_id) for c in ContractRegistry.default().contracts()}
        assert set(COSTS) | MEMORY_BOUND <= names
        assert not set(COSTS) & MEMORY_BOUND

    def test_a_contract_without_a_formula_moves_its_tensors_inexactly(self) -> None:
        for case in OpCase.all():
            cost = case.contract.cost_at(case.point)
            assert cost.bytes_moved > 0, case.contract_id
            name = contract_name(case.contract_id)
            assert cost.exact == (name in COSTS or name in MEMORY_BOUND), case.contract_id
            assert name in COSTS or (cost.flops, cost.flops_dtype) == (0, None), case.contract_id
