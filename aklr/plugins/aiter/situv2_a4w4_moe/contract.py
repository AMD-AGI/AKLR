# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    GateUpInterleavedScales,
    GateUpInterleavedWeight,
    IntRangeDistribution,
    LogUniformDistribution,
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

__all__ = ["CONTRACTS", "SITUV2_A4W4_MOE"]

_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))
_FINITE = FiniteDistribution()


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _facts(shuffled: bool, scale_rank: int) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": scale_rank}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


def _beta(description: str) -> ScalarSpec:
    return ScalarSpec(
        description=description,
        dtype=DType.FLOAT32,
        interval=(2.0**-126, DTYPE_TRAITS["float32"].largest_finite_value),
    )


SITUV2_A4W4_MOE = OpContract(
    contract_id="situv2_a4w4_moe@1.0",
    description=(
        "aiter's fused MoE as vLLM calls it for SiTUv2 experts with MXFP4 weights and MXFP4 activations "
        "(VLLM_ROCM_USE_AITER_MOE_SITUV2 a4w4, AITER_SITUV2_A4W4): each token through its top-k experts, "
        "down(situ(gate(x)) * linear(up(x))) of fp4 weights at an e8m0 scale per 32, each stage's activations "
        "rounded to MXFP4, weighted by its routing weight, summed. w1 keeps the gate rows before the up rows; "
        "aiter's a16w4 shuffle, gate and up not interleaved, holds both weights and both scales."
    ),
    op_type="situv2_a4w4_moe",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_experts": axis("Experts.", 1, 1024),
        "topk": axis("Experts per token.", 1, 32),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    constraints=("topk <= n_experts",),
    inputs={
        "hidden_states": TensorSpec(
            description="Token activations.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=_FINITE,
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "topk_weights": TensorSpec(
            description="Routing weight of each chosen expert.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=_FINITE,
        ),
        "topk_ids": TensorSpec(
            description="Chosen experts per token, distinct within a row.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo="0", hi="n_experts - 1", distinct=True),
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("model_dim"),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "beta": _beta("SiTUv2 gate saturation."),
        "linear_beta": _beta("SiTUv2 up saturation."),
    },
    outputs={
        "out": TensorSpec(
            description="Routing-weighted sum of the chosen experts' outputs.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            # fused_moe's bound: its MXFP4 activations
            error_bounds=SqnrBound(min_sqnr_db=13.0),
        )
    },
    # as vLLM holds the experts for aiter's a4w4 (shuffle_weight_a16w4 and shuffle_scale_a16w4 with gate_up
    # False, e8m0_shuffle for w2's scales): the same bytes as the gate-up layouts without interleave
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(layouts={}, detect_facts=_facts(False, 3)),
            "a16w4_shuffled": LayoutVariant(
                layouts={
                    "w1": GateUpInterleavedWeight(gate_up=False),
                    "w2": GateUpInterleavedWeight(gate_up=False),
                    "w1_scale": GateUpInterleavedScales(gate_up=False),
                    "w2_scale": GateUpInterleavedScales(gate_up=False),
                },
                detect_facts=_facts(True, 2),
            ),
        }
    ),
    # stage 2 combines with atomics
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (SITUV2_A4W4_MOE,)
