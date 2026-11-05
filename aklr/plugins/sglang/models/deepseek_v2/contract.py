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

__all__ = ["CONTRACTS", "DEEPSEEK_V2_FP8_MOE", "DEEPSEEK_V2_MXFP4_MOE"]

_SAME_AS_HIDDEN = DTypeSpec(same_as_tensor="hidden_states")
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))

_ROUTING = (
    "on hidden_states, each token's topk experts by sigmoid score plus "
    "correction_bias, the scores those of the router's logits in hidden_states' dtype (as "
    "sglang's MoEGate returns them on ROCm), weighed by the renormalized unbiased scores "
    "times routed_scaling_factor, and the shared expert at weight 1, each expert "
    "down(silu(gate(x)) * up(x))"
)
_RANK = "out is their output on this rank, which sglang sums over the ranks."


def _axes(blocks: dict) -> dict:
    """The axes both contracts share, then the scale ``blocks`` of their format."""
    return {
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        **blocks,
    }


_ATTRS = {
    "topk": attr("Routed experts per token.", 8, param_name="topk"),
    "num_expert_group": attr("Expert groups; one, ungrouped.", 1, param_name="num_expert_group"),
    "topk_group": attr("Groups kept; one, ungrouped.", 1, param_name="topk_group"),
    "renormalize": attr(
        "Whether the routed weights are divided by their sum.",
        True,
        param_name="renormalize",
    ),
    "scoring_func": attr("Router score.", "sigmoid", param_name="scoring_func"),
}
_ROUTING_INPUTS = {
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
        dtype=_FLOAT32,
        # about 8, a spread like GLM-5's
        distribution=LogUniformDistribution(log2_lo="2.99", log2_hi="3.01"),
        lifetime="weight",
    ),
}
_ROUTED_SCALING_FACTOR = ScalarSpec(
    description="Multiplies the routed weights only: GLM-5's.",
    dtype=DType.FLOAT32,
    values=(2.5,),
    default=2.5,
)


def _output(min_sqnr_db: float) -> TensorSpec:
    return TensorSpec(
        description="The experts' output on this rank, before sglang's all-reduce sums the ranks'.",
        shape=("n_tokens", "model_dim"),
        dtype=_SAME_AS_HIDDEN,
        error_bounds=SqnrBound(min_sqnr_db=min_sqnr_db),
    )


_COMPARE = CompareSpec(determinism=DeterminismLevel.BOUNDED)


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
        f"last slot: {_ROUTING} of its MXFP4 weights (fused_moe's: fp4 values at an e8m0 "
        f"scale per 32); {_RANK}"
    ),
    op_type="deepseek_v2",
    axes=_axes(
        {
            "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
            "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
        }
    ),
    attrs=_ATTRS,
    constraints=(
        "topk <= n_routed",
        # makes the scales' padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        **_ROUTING_INPUTS,
        **_MXFP4_EXPERTS,
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    # the experts' MXFP4 activations alone come to about 12 dB
    outputs={"out": _output(10.0)},
    layout_variants=_MXFP4_LAYOUTS,
    compare=_COMPARE,
)

_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
# as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")

# fp8 experts as sglang's Fp8MoEMethod holds them on gfx950: e4m3fn at a scale per 128 x 128 block
_FP8_EXPERTS = {
    "w1": TensorSpec(
        description="Gate-up projection per expert: gate rows, then up rows.",
        shape=("n_experts", "gate_up_rows", "model_dim"),
        dtype=_FP8,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "w1_scale": TensorSpec(
        description="w1's dequant scale per 128 x 128 block.",
        shape=("n_experts", "gate_up_blocks", "model_blocks"),
        dtype=_FLOAT32,
        distribution=_BLOCK_SCALE,
        lifetime="weight",
    ),
    "w2": TensorSpec(
        description="Down projection per expert.",
        shape=("n_experts", "model_dim", "inter_dim"),
        dtype=_FP8,
        distribution=LecunNormalDistribution(fan_in="inter_dim"),
        lifetime="weight",
    ),
    "w2_scale": TensorSpec(
        description="w2's dequant scale per 128 x 128 block.",
        shape=("n_experts", "model_blocks", "inter_blocks"),
        dtype=_FLOAT32,
        distribution=_BLOCK_SCALE,
        lifetime="weight",
    ),
}


def _fp8_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    return tuple(
        LayoutFact(location="tensor_attr", name=f"{w}.is_shuffled", expected_value=shuffled)
        for w in ("w1", "w2")
    )


# row-major, or as sglang holds them for aiter: shuffle_weight's 16 x 16 tiles, the scales as the
# checkpoint stores them
_FP8_LAYOUTS = LayoutVariantsSpec(
    by_name={
        "row_major": LayoutVariant(detect_facts=_fp8_facts(False)),
        "shuffled_16x16": LayoutVariant(
            layouts=dict.fromkeys(("w1", "w2"), Preshuffle(tile=(16, 16))),
            detect_facts=_fp8_facts(True),
        ),
    }
)

DEEPSEEK_V2_FP8_MOE = OpContract(
    contract_id="deepseek_v2_fp8_moe@1.0",
    description=(
        "sglang's DeepseekV2MoE on one rank, with fp8 experts and the shared expert in their "
        f"last slot: {_ROUTING} of its fp8 weights at a scale per 128 x 128 block, each of its "
        "two linears taking its input as the checkpoint's FP8 scheme quantizes it: in bf16, "
        f"then each 1 x 128 group in fp8 at a scale of its amax / 448; {_RANK}"
    ),
    op_type="deepseek_v2",
    axes=_axes(
        {
            "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
            "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
            "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
        }
    ),
    attrs=_ATTRS,
    constraints=("topk <= n_routed",),
    inputs={
        **_ROUTING_INPUTS,
        **_FP8_EXPERTS,
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    # fp8 alone comes to about 36 dB; a near tie of the 8th and 9th scores may pick the other expert
    outputs={"out": _output(20.0)},
    layout_variants=_FP8_LAYOUTS,
    compare=_COMPARE,
)

CONTRACTS = (DEEPSEEK_V2_FP8_MOE, DEEPSEEK_V2_MXFP4_MOE)
