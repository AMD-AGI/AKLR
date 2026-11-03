# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    E8M0Scales,
    FiniteDistribution,
    LecunNormalDistribution,
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

__all__ = ["CONTRACTS", "DEEPSEEK_V2_MXFP4_MOE"]

_SAME_AS_HIDDEN = DTypeSpec(same_as_tensor="hidden_states")


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _mxfp4_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": 3}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


# MXFP4 experts as aiter's fused_moe takes them: fp4 values at an e8m0 scale per 32
_MXFP4_EXPERTS = {
    "w1": TensorSpec(
        description="Gate-up projection per expert: gate rows, then up rows.",
        shape=("n_experts", "gate_up_rows", "model_dim"),
        dtype=DTypeSpec(any_of=(DType.FLOAT4_E2M1,)),
        distribution=FiniteDistribution(),
        lifetime="weight",
    ),
    "w1_scale": TensorSpec(
        description="Per 1 x 32 block of w1.",
        shape=("n_experts", "gate_up_rows", "model_blocks"),
        dtype=DTypeSpec(any_of=(DType.FLOAT8_E8M0,)),
        distribution=_e8m0_scales("model_dim"),
        lifetime="weight",
    ),
    "w2": TensorSpec(
        description="Down projection per expert.",
        shape=("n_experts", "model_dim", "inter_dim"),
        dtype=DTypeSpec(any_of=(DType.FLOAT4_E2M1,)),
        distribution=FiniteDistribution(),
        lifetime="weight",
    ),
    "w2_scale": TensorSpec(
        description="Per 1 x 32 block of w2.",
        shape=("n_experts", "model_dim", "inter_blocks"),
        dtype=DTypeSpec(any_of=(DType.FLOAT8_E8M0,)),
        distribution=_e8m0_scales("inter_dim"),
        lifetime="weight",
    ),
}
# row-major, or as sglang holds them for aiter: shuffle_weight's 16 x 16 tiles, the scales shuffled
_MXFP4_LAYOUTS = LayoutVariantsSpec(
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
)

DEEPSEEK_V2_MXFP4_MOE = OpContract(
    contract_id="deepseek_v2_mxfp4_moe@1.0",
    description=(
        "sglang's DeepseekV2MoE on one rank, with MXFP4 experts and the shared expert in their "
        "last slot: on hidden_states, each token's topk experts by sigmoid score plus "
        "correction_bias, the scores those of the router's logits in hidden_states' dtype (as "
        "sglang's MoEGate returns them on ROCm), weighed by the renormalized unbiased scores "
        "times routed_scaling_factor, and the shared expert at weight 1, each expert "
        "down(silu(gate(x)) * up(x)) of its MXFP4 weights (fused_moe's: fp4 values at an e8m0 "
        "scale per 32); out is their output on this rank, which sglang sums over the ranks."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs={
        "topk": attr("Routed experts per token.", 8, param_name="topk"),
        "num_expert_group": attr("Expert groups; one, ungrouped.", 1, param_name="num_expert_group"),
        "topk_group": attr("Groups kept; one, ungrouped.", 1, param_name="topk_group"),
        "renormalize": attr(
            "Whether the routed weights are divided by their sum.",
            True,
            param_name="renormalize",
        ),
        "scoring_func": attr("Router score.", "sigmoid", param_name="scoring_func"),
    },
    constraints=(
        "topk <= n_routed",
        # makes the scales' padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        "hidden_states": TensorSpec(
            description="The MoE's input: post_attention_layernorm's output.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
        "router_weight": TensorSpec(
            description="Router projection.",
            shape=("n_routed", "model_dim"),
            dtype=_SAME_AS_HIDDEN,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "correction_bias": TensorSpec(
            description="Added to the scores to pick experts; never weighs them.",
            shape=("n_routed",),
            # fp32 as GLM-5 stores it and sglang keeps it
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            # about 8, a spread like GLM-5's
            distribution=LogUniformDistribution(log2_lo="2.99", log2_hi="3.01"),
            lifetime="weight",
        ),
        **_MXFP4_EXPERTS,
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies the routed weights only: GLM-5's.",
            dtype=DType.FLOAT32,
            values=(2.5,),
            default=2.5,
        ),
    },
    outputs={
        "out": TensorSpec(
            description=("The experts' output on this rank, before sglang's all-reduce sums the ranks'."),
            shape=("n_tokens", "model_dim"),
            dtype=_SAME_AS_HIDDEN,
            # the experts' MXFP4 activations alone come to about 12 dB
            error_bounds=SqnrBound(min_sqnr_db=10.0),
        )
    },
    layout_variants=_MXFP4_LAYOUTS,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (DEEPSEEK_V2_MXFP4_MOE,)
