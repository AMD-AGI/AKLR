# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    E8M0Scales,
    ExpertMaskDistribution,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "MUSE_SPARK_MOE"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _mxfp4_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": 3}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


MUSE_SPARK_MOE = OpContract(
    contract_id="muse_spark_moe@1.0",
    description=(
        "vLLM's MuseSpark1xMoE experts on one expert-parallel rank with MXFP4 experts, from "
        "the pooled rows to the unweighted output of each route: each row's pre-expert "
        "RMSNorm (fp32, times norm_weight + gain_center), then per route, the routes to the "
        "experts expert_mask gives this rank, down(silu(gate(x)) * up(x)) of its MXFP4 "
        "weights (fused_moe's: fp4 values at an e8m0 scale per 32); the routes to the others "
        "write zeros. The router weights, the post-expert norm and the sum over the routes "
        "are vLLM's, after it."
    ),
    op_type="muse_spark",
    axes={
        "n_tokens": axis("Pooled rows in this call.", 1, 65536, is_var=True),
        "model_dim": axis("The pooled width: the experts' hidden size.", 2048, 16384, multiple_of=2048),
        "topk": axis("Routes per row.", 1, 32),
        "n_experts": axis("Experts this rank holds: w1's and w2's.", 2, 1024),
        "inter_dim": axis("Expert intermediate size.", 256, 16384, multiple_of=256),
        "global_experts": axis(
            "expert_mask's length: the group's routed experts, then vLLM's always-masked slot.",
            3,
            1025,
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    # vLLM's expert map gives each rank an equal block of the routed experts
    constraints=("topk < global_experts", "(global_experts - 1) % n_experts == 0"),
    inputs={
        "hidden_states": TensorSpec(
            description="The pooled rows, not yet normalized.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "norm_weight": TensorSpec(
            description="The pre-expert RMSNorm's gain, before gain_center is added.",
            shape=("model_dim",),
            dtype=_BF16,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "topk_ids": TensorSpec(
            description="Each row's routed experts, distinct within a row: global ids.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo="0", hi="global_experts - 2", distinct=True),
        ),
        "expert_mask": TensorSpec(
            description="1 for the experts this rank holds, in the order w1 holds them; 0 in the last slot.",
            shape=("global_experts",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=ExpertMaskDistribution(local_experts="n_experts"),
            lifetime="weight",
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("model_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "eps": ScalarSpec(
            description="The RMSNorm's epsilon.",
            dtype=DType.FLOAT32,
            interval=(1e-12, 1e-3),
            default=1e-6,
        ),
        "gain_center": ScalarSpec(
            description="Added to norm_weight: 1 for a zero-centered gain.",
            dtype=DType.FLOAT32,
            values=(0.0, 1.0),
            default=1.0,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="Each route's expert output, unweighted; zeros off this rank.",
            shape=("n_tokens", "topk", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            # fused_moe's bound: its MXFP4 activations
            error_bounds=SqnrBound(min_sqnr_db=13.0),
        )
    },
    # as vLLM holds the experts for aiter: shuffle_weight's 16 x 16 tiles, the scales shuffled
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(layouts={}, detect_facts=_mxfp4_facts(False)),
            "shuffled_16x16": LayoutVariant(
                layouts={
                    "w1": Preshuffle(tile=(16, 16)),
                    "w2": Preshuffle(tile=(16, 16)),
                    "w1_scale": E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True),
                    "w2_scale": E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True),
                },
                detect_facts=_mxfp4_facts(True),
            ),
        }
    ),
    # stage 2 combines with atomics
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (MUSE_SPARK_MOE,)
