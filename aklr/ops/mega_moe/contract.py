# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.ops.fused_moe.contract import FUSED_MOE

__all__ = ["CONTRACTS", "MEGA_MOE"]

_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))
_FINITE_DISTRIBUTION = FiniteDistribution()


def _e8m0_scale_distribution(fan_in: str) -> LogUniformDistribution:
    """fused_moe's silu scale draw: one per 1 x 32 block, around 1 / sqrt(fan-in)."""
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2",
        log2_hi=f"1 - bit_length({fan_in}) // 2",
    )


MEGA_MOE = OpContract(
    contract_id="mega_moe@1.0",
    description=(
        "Expert-parallel MoE in one call on every rank of the expert-parallel group:"
        " each token of this rank goes to its top-k experts, held across the group's"
        " ranks (rank r holds global experts r * n_experts to (r + 1) * n_experts - 1),"
        " and out is the routing-weighted sum of their outputs, back on this rank."
        " An expert is down(silu(min(gate, limit)) * clamp(up, -limit, limit)) of its"
        " MXFP4 weights, each linear taking its input in MXFP8 (1 x 32 blocks, E8M0"
        " scales). Every rank of the group calls the op once per step, in the same order."
    ),
    op_type="mega_moe",
    axes={
        "n_tokens": axis("Tokens of this rank in this call.", 1, 32768, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_experts": axis("Experts on each rank.", 1, 1024),
        "topk": axis("Experts per token.", 1, 16),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs={
        "ep_size": attr(
            "Ranks of the expert-parallel group.",
            1,
            2,
            4,
            8,
            16,
            default=1,
            param_name="ep_size",
        ),
    },
    constraints=(
        "topk <= n_experts * ep_size",
        # the kernels take the experts only as sglang's MegaMoE adapter holds them
        "layout_variant == 'gate_up_interleaved'",
    ),
    inputs={
        "hidden_states": TensorSpec(
            description="Token activations of this rank.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "w1": TensorSpec(
            description="Gate-up projection of this rank's experts: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection of this rank's experts.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "topk_weights": TensorSpec(
            description="Routing weight of each chosen expert.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "topk_ids": TensorSpec(
            description="Chosen global experts per token, distinct within a row.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo="0", hi="n_experts * ep_size - 1", distinct=True),
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scale_distribution("model_dim"),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scale_distribution("inter_dim"),
            lifetime="weight",
        ),
        "swiglu_limit": ScalarSpec(
            description="Clamped SwiGLU bound: gate at most this, up within +/- this.",
            dtype=DType.FLOAT32,
            values=(10.0,),
            default=10.0,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="Routing-weighted sum of the chosen experts' outputs.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        )
    },
    # how sglang's MegaMoE adapter holds the experts (aiter shuffle_weight_a16w4 and
    # shuffle_scale_a16w4)
    layout_variants=LayoutVariantsSpec(
        by_name={n: FUSED_MOE.layout_variants.by_name[n] for n in ("row_major", "gate_up_interleaved")}
    ),
    # Bounded, not bitwise: the combine adds the top-k outputs in arrival order.
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
    ranks="ep_size",
)

CONTRACTS = (MEGA_MOE,)
