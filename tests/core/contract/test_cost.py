# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect

import pytest

from aklr.core.contract.cost import Cost
from aklr.core.contract.locate import op_module
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import contract_name
from aklr.ops.fused_moe.cost import fused_moe_cost
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
        assert fused_moe_cost(symbols, {"w1": 0, "w2": 0, "w1_scale": 0, "w2_scale": 0}).dtype == dtype

    def test_norm_is_bytes_alone(self) -> None:
        case = OpCase.of("rms_norm@1.0")
        assert case.contract.cost_at(case.point) == Cost(0, (2 * 4 * 2048 + 2048) * 2, None)

    def test_every_cost_function_is_a_contracts(self) -> None:
        """``cost_at`` finds ``<contract name>_cost`` by name: one named after no contract of
        its op is a misspelling that leaves the contract without a cost."""
        contracts = ContractRegistry.default().contracts()
        named = {(c.op_package, f"{contract_name(c.contract_id)}_cost") for c in contracts}
        for package in {c.op_package for c in contracts}:
            try:
                module = op_module(package, "cost")
            except ModuleNotFoundError:
                continue
            for name, _ in inspect.getmembers(module, inspect.isfunction):
                if name.endswith("_cost"):
                    assert (package, name) in named, f"{module.__name__}.{name}"

    def test_an_op_without_a_cost_function_has_none(self) -> None:
        case = OpCase.of("unified_attention@1.0")
        assert case.contract.cost_at(case.point) is None
