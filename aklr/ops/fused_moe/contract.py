# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BaselineRelativeBound,
    DeterminismLevel,
    Distribution,
    DType,
    E8M0Scales,
    ExpertMaskDistribution,
    FiniteDistribution,
    GateUpInterleavedScales,
    GateUpInterleavedWeight,
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
    attr,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "FUSED_MOE"]


_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))


def _beta(description: str) -> ScalarSpec:
    return ScalarSpec(
        description=description,
        dtype=DType.FLOAT32,
        interval=(2.0**-126, DTYPE_TRAITS["float32"].largest_finite_value),
        present_when="activation == 'situv2'",
    )


_FINITE_DISTRIBUTION = FiniteDistribution()


def _e8m0_scale_distribution(fan_in: str, shift: int = 0) -> Distribution:
    return LogUniformDistribution(
        log2_lo=f"{shift - 5} - bit_length({fan_in}) // 2",
        log2_hi=f"{shift + 1} - bit_length({fan_in}) // 2",
    )


def _variant_facts(shuffled: bool, scale_rank: int) -> tuple[LayoutFact, ...]:
    facts = {
        "w1.is_shuffled": shuffled,
        "w2.is_shuffled": shuffled,
        "w1_scale.ndim": scale_rank,
    }
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


# The same scales as gemm_a4w4's; inside the domain the padding is a no-op.
_SHUFFLED_SCALES = E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)
_PRESHUFFLE = Preshuffle(tile=(16, 16))

FUSED_MOE = OpContract(
    contract_id="fused_moe@1.0",
    description="Route each token to its top-k experts, run both projections, combine.",
    op_type="fused_moe",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_experts": axis("Local experts on this rank.", 1, 1024),
        "topk": axis("Experts per token.", 1, 32),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "global_experts": axis("Experts of the whole expert-parallel group.", 1, 1024),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs={
        "activation": attr(
            "Gate activation.",
            "silu",
            "gelu",
            "situv2",
            "swiglu",
            default="silu",
            param_name="activation",
        ),
        "quant_type": attr("Weight quantization.", "per_1x32", param_name="quant_type"),
        "doweight_stage1": attr(
            "Whether the routing weight is applied in stage 1.",
            False,
            True,
            default=False,
            param_name="doweight_stage1",
        ),
        "gate_mode": attr(
            "Where w1 keeps the up rows: after all gate rows, or interleaved with"
            " them (aiter's clamped SwiGLU, DeepSeek-V4).",
            "separated",
            "interleave",
            default="separated",
            param_name="gate_mode",
        ),
        "has_expert_mask": attr("Whether expert_mask is passed.", False, True, default=False),
        "has_num_local_tokens": attr(
            "Whether num_local_tokens is passed.",
            False,
            True,
            default=False,
            optional=True,
        ),
        "has_swiglu_limit": attr(
            "Whether swiglu_limit is passed.", False, True, default=False, optional=True
        ),
        "act_quant": attr(
            "Bits per quantized activation element; gate_mode 'interleave' keeps"
            " bf16 (aiter's fp8 past AITER_BF16_FP8_MOE_BOUND rows is in the bound),"
            " swiglu keeps bf16 below 256 rows (aiter's"
            " GPTOSS_SWIGLU_MXFP4_BF16_BOUND).",
            4,
            8,
            default=4,
            param_name="act_quant",
        ),
    },
    constraints=(
        "has_expert_mask or topk <= n_experts",
        # aiter masks experts off this rank for silu a4w4 too (vLLM's EP); its a4w4 has no gelu.
        # Expert parallelism is also described for the clamped SwiGLU and swiglu.
        (
            "(not has_expert_mask) or activation == 'silu' or activation == 'situv2'"
            " or gate_mode == 'interleave' or activation == 'swiglu'"
        ),
        "(not has_expert_mask) or topk <= global_experts",
        "(not has_expert_mask) or n_experts <= global_experts",
        # Legacy describes situv2 as a8w4 alone, every other activation as a4w4.
        "activation == 'situv2' or act_quant == 4",
        "activation != 'situv2' or act_quant == 8",
        "activation != 'situv2' or layout_variant == 'gate_up_interleaved'",
        ("activation == 'situv2' or gate_mode == 'interleave' or layout_variant != 'gate_up_interleaved'"),
        # aiter's clamped SwiGLU: silu on sglang's gate-up-interleaved weights
        ("gate_mode == 'separated' or (activation == 'silu' and layout_variant == 'gate_up_interleaved')"),
        "has_swiglu_limit == (gate_mode == 'interleave' or activation == 'swiglu')",
        # aiter's swiglu path is measured on preshuffled weights, weighted in stage 2
        ("activation != 'swiglu' or (layout_variant == 'shuffled_16x16' and (not doweight_stage1))"),
        # aiter's bf16 swiglu splits model_dim in two, each half whole 256-wide tiles
        "activation != 'swiglu' or model_dim % 512 == 0",
        "(not has_num_local_tokens) or (has_expert_mask and gate_mode == 'interleave')",
        # Legacy's unpadded-scale check; it makes aiter's scale padding a no-op.
        (
            "activation == 'situv2' or gate_mode == 'interleave'"
            " or (model_dim % 256 == 0 and inter_dim % 256 == 0)"
        ),
    ),
    inputs={
        "hidden_states": TensorSpec(
            description="Token activations.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=_FINITE_DISTRIBUTION,
            lifetime="weight",
        ),
        "topk_weights": TensorSpec(
            description="Routing weight of each chosen expert.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32, DType.BFLOAT16, DType.FLOAT16)),
            distribution=_FINITE_DISTRIBUTION,
        ),
        "topk_ids": TensorSpec(
            description=(
                "Chosen experts per token, distinct within a row; global ids under expert parallelism."
            ),
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32, DType.INT64)),
            distribution=(
                (
                    "has_expert_mask",
                    IntRangeDistribution(lo="0", hi="global_experts - 1", distinct=True),
                ),
                (
                    None,
                    IntRangeDistribution(lo="0", hi="n_experts - 1", distinct=True),
                ),
            ),
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            # 4x larger for swiglu, so that its clamp at 7 binds on some elements
            distribution=(
                ("activation == 'swiglu'", _e8m0_scale_distribution("model_dim", 2)),
                (None, _e8m0_scale_distribution("model_dim")),
            ),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scale_distribution("inter_dim"),
            lifetime="weight",
        ),
        "expert_mask": TensorSpec(
            description="1 for the experts this rank holds, in the order w1 holds them.",
            shape=("global_experts",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            present_when="has_expert_mask",
            distribution=ExpertMaskDistribution(local_experts="n_experts"),
            lifetime="weight",  # fixed per rank
        ),
        "num_local_tokens": TensorSpec(
            description="Rows this rank received; later rows are unused.",
            shape=(1,),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            present_when="has_num_local_tokens",
            distribution=IntRangeDistribution(lo="n_tokens", hi="n_tokens"),
        ),
        "beta": _beta("SiTUv2 gate saturation."),
        "linear_beta": _beta("SiTUv2 up saturation."),
        "swiglu_limit": ScalarSpec(
            description="Clamped SwiGLU bound: gate at most this, up within +/- this.",
            dtype=DType.FLOAT32,
            # 7.0 is gpt-oss swiglu's (aiter's swiglu_and_mul fixes it); the clamped
            # SwiGLU (gate_mode 'interleave') is validated from DeepSeek-V4's 10 only:
            # aiter's CK-Tile decode path does not clamp, so an interleave limit below
            # 10 that binds on the bench's inputs has no baseline
            interval=(7.0, DTYPE_TRAITS["float32"].largest_finite_value),
            present_when="has_swiglu_limit",
        ),
    },
    outputs={
        "out": TensorSpec(
            description=(
                "Routing-weighted sum of the chosen experts' outputs;"
                " under expert parallelism, this rank's share."
            ),
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=(
                # aiter's a16 and a4 swiglu errors differ by orders of magnitude: the
                # bound follows the baseline's
                ("activation == 'swiglu'", BaselineRelativeBound()),
                # FIXME(main-kernels): main gates fused_moe on output-dtype SQNR >= 13
                # dB; its e33/e65 kernels keep fp8/MXFP4 partials, far off aiter's error.
                (None, SqnrBound(min_sqnr_db=13.0)),
            ),
        )
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(layouts={}, detect_facts=_variant_facts(False, 3)),
            "shuffled_16x16": LayoutVariant(
                layouts={
                    "w1": _PRESHUFFLE,
                    "w2": _PRESHUFFLE,
                    "w1_scale": _SHUFFLED_SCALES,
                    "w2_scale": _SHUFFLED_SCALES,
                },
                detect_facts=_variant_facts(True, 3),
            ),
            "gate_up_interleaved": LayoutVariant(
                layouts={
                    "w1": GateUpInterleavedWeight(gate_up=True),
                    "w2": GateUpInterleavedWeight(gate_up=False),
                    "w1_scale": GateUpInterleavedScales(gate_up=True),
                    "w2_scale": GateUpInterleavedScales(gate_up=False),
                },
                detect_facts=_variant_facts(True, 2),
            ),
        }
    ),
    # Bounded, not bitwise: stage 2 often combines top-k with atomics (design §5.4).
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (FUSED_MOE,)
