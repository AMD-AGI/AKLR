# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DsaIndexKCacheDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    AttrSpec,
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
from aklr.ops.fused_moe.contract import FUSED_MOE

__all__ = [
    "CONTRACTS",
    "DEEPSEEK_V2_ATTENTION",
    "DEEPSEEK_V2_ATTENTION_INDEXER",
    "DEEPSEEK_V2_DENSE_LAYER",
    "DEEPSEEK_V2_EH_PROJ",
    "DEEPSEEK_V2_FP8_ATTENTION_INDEXER",
    "DEEPSEEK_V2_FP8_DENSE_LAYER",
    "DEEPSEEK_V2_FP8_MOE_LAYER",
    "DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART",
    "DEEPSEEK_V2_MOE",
    "DEEPSEEK_V2_MOE_LAYER",
    "DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER",
    "DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_SAME_AS_HIDDEN = DTypeSpec(same_as_tensor="hidden_states")
_FINITE = FiniteDistribution()
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_POSITIVE_FLOAT32 = (2.0**-149, _FLOAT32_LARGEST)
# GLM-5's norm epsilon and softmax scale, 1 / sqrt(nope_dim + rope_dim): the defaults bench's
# inputs take
_EPS = 1e-5
_SM_SCALE = 0.0625
# a checkpoint's attention weights: fp8 with dequant scales, or bf16 without
_WEIGHT_DTYPES = (DType.FLOAT8_E4M3FN, DType.BFLOAT16)
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
_COMPARE = CompareSpec(determinism=DeterminismLevel.BOUNDED)


def _has(scale: str) -> str:
    """The presence attr of ``scale``."""
    return f"has_{scale}"


def _scale_attr(scale: str, weight: str, first: bool) -> AttrSpec:
    """The presence attr of ``scale``, which fp8 ``weight`` takes; ``first`` its first value."""
    return attr(f"Whether {scale} is passed: {weight} is fp8.", first, not first, abbrev=scale)


def _scale_constraint(scale: str, weight: str) -> str:
    return f"{_has(scale)} == ({weight}.dtype == 'float8_e4m3fn')"


def _shuffled(*weights: str) -> LayoutVariantsSpec:
    """Row-major weights, or all of them as sglang holds them for aiter: shuffle_weight's
    16 x 16 tiles."""

    def facts(shuffled: bool) -> tuple[LayoutFact, ...]:
        return tuple(
            LayoutFact(location="tensor_attr", name=f"{w}.is_shuffled", expected_value=shuffled)
            for w in weights
        )

    return LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=facts(False)),
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(weights, Preshuffle(tile=(16, 16))),
                detect_facts=facts(True),
            ),
        }
    )


def _block_scale(rows: str, cols: str, present_when: str | None = None) -> TensorSpec:
    """An fp8 weight's scale per 128 x 128 block [rows, cols], passed when attr ``present_when``
    is (if given)."""
    return TensorSpec(
        description="An fp8 weight's dequant scale per 128 x 128 block.",
        shape=(rows, cols),
        dtype=_FLOAT32,
        present_when=present_when,
        distribution=_BLOCK_SCALE,
        lifetime="weight",
    )


def _linear(
    name: str,
    description: str,
    rows: str,
    cols: str,
    scale_rows: str,
    scale_cols: str,
    same_as: str | None = None,
) -> dict[str, TensorSpec]:
    """Linear ``name``: its weight [rows, cols], fp8 or bf16 (as ``same_as``'s, if given), and
    an fp8 weight's scale per 128 x 128 block [scale_rows, scale_cols] (presence attr
    ``has_<name>_scale``)."""
    return {
        f"{name}_weight": TensorSpec(
            description=description,
            shape=(rows, cols),
            dtype=DTypeSpec(any_of=_WEIGHT_DTYPES, same_as_tensor=same_as),
            distribution=LecunNormalDistribution(fan_in=cols),
            lifetime="weight",
        ),
        f"{name}_scale": _block_scale(scale_rows, scale_cols, _has(f"{name}_scale")),
    }


# The post_attention_layernorm the MoE and dense layer contracts start with
_POST_NORM = {
    "hidden_states": TensorSpec(
        description="The residual stream after the attention.",
        shape=("n_tokens", "model_dim"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "norm_weight": TensorSpec(
        description="post_attention_layernorm's weight.",
        shape=("model_dim",),
        dtype=_SAME_AS_HIDDEN,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "norm_eps": ScalarSpec(
        description="post_attention_layernorm's epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=_EPS,
    ),
}


def _layer_output(min_sqnr_db: float) -> TensorSpec:
    return TensorSpec(
        description="The residual stream after the layer.",
        shape=("n_tokens", "model_dim"),
        dtype=_SAME_AS_HIDDEN,
        error_bounds=SqnrBound(min_sqnr_db=min_sqnr_db),
    )


# DeepseekV2MoE's routing, as both MoE layer contracts take it: each token's topk experts by
# sigmoid score plus correction_bias, weighed by the renormalized unbiased scores times
# routed_scaling_factor
_ROUTING_ATTRS = {
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
    description="Multiplies the routed weights only.",
    dtype=DType.FLOAT32,
    interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
    default=1.0,
)

DEEPSEEK_V2_MOE = OpContract(
    contract_id="deepseek_v2_moe@1.0",
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
    attrs=_ROUTING_ATTRS,
    constraints=(
        "topk <= n_routed",
        # fused_moe's: it makes aiter's scale padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        "hidden_states": TensorSpec(
            description="The MoE's input: post_attention_layernorm's output.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        **_ROUTING_INPUTS,
        # fused_moe's MXFP4 experts, as aiter takes them; its distributions' last
        # branches, the silu ones (its predicates read attrs this layer lacks)
        **{
            n: FUSED_MOE.inputs[n].model_copy(update={"distribution": FUSED_MOE.inputs[n].distribution[-1:]})
            for n in ("w1", "w1_scale", "w2", "w2_scale")
        },
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
    layout_variants=LayoutVariantsSpec(
        by_name={n: FUSED_MOE.layout_variants.by_name[n] for n in ("row_major", "shuffled_16x16")}
    ),
    compare=_COMPARE,
)

DEEPSEEK_V2_MOE_LAYER = OpContract(
    contract_id="deepseek_v2_moe_layer@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention, on one rank, with MXFP4 experts: "
        "the residual stream hidden_states through post_attention_layernorm, then on it each "
        "token's topk experts by sigmoid score plus correction_bias, weighed by the renormalized "
        "unbiased scores times routed_scaling_factor, and the shared expert in the last slot at "
        "weight 1, each expert down(silu(gate(x)) * up(x)) of its MXFP4 weights (fused_moe's: fp4 "
        "values at an e8m0 scale per 32); out is hidden_states plus their output summed over the "
        "ranks of the kernel's peer memory in rank order, the stream after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs=_ROUTING_ATTRS,
    constraints=(
        "topk <= n_routed",
        # fused_moe's: it makes aiter's scale padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        **_POST_NORM,
        **_ROUTING_INPUTS,
        # fused_moe's MXFP4 experts, as aiter takes them; its distributions' last
        # branches, the silu ones (its predicates read attrs this layer lacks)
        **{
            n: FUSED_MOE.inputs[n].model_copy(update={"distribution": FUSED_MOE.inputs[n].distribution[-1:]})
            for n in ("w1", "w1_scale", "w2", "w2_scale")
        },
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    # the experts' MXFP4 activations alone come to about 12 dB
    outputs={"out": _layer_output(10.0)},
    layout_variants=LayoutVariantsSpec(
        by_name={n: FUSED_MOE.layout_variants.by_name[n] for n in ("row_major", "shuffled_16x16")}
    ),
    compare=_COMPARE,
)

DEEPSEEK_V2_FP8_MOE_LAYER = OpContract(
    contract_id="deepseek_v2_fp8_moe_layer@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention, on one rank, with fp8 experts: "
        "the residual stream hidden_states through post_attention_layernorm, then on it each "
        "token's topk experts by sigmoid score plus correction_bias, weighed by the renormalized "
        "unbiased scores times routed_scaling_factor, and the shared expert in the last slot at "
        "weight 1, each expert down(silu(gate(x)) * up(x)) of its fp8 weights at a scale per "
        "128 x 128 block, each of its two linears taking its input as the checkpoint's FP8 "
        "scheme quantizes it: in bf16, then each 1 x 128 group in fp8 at a scale of its "
        "amax / 448; out is hidden_states plus their output summed over the ranks of the "
        "kernel's peer memory in rank order, the stream after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
        "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
    },
    attrs=_ROUTING_ATTRS,
    constraints=("topk <= n_routed",),
    inputs={
        **_POST_NORM,
        **_ROUTING_INPUTS,
        "w1": TensorSpec(
            description="Gate-up projection per expert slot: gate rows, then up rows.",
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
            description="Down projection per expert slot.",
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
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    outputs={"out": _layer_output(40.0)},
    # sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them, the
    # scales as the checkpoint stores them
    layout_variants=_shuffled("w1", "w2"),
    compare=_COMPARE,
)

DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART = OpContract(
    contract_id="deepseek_v2_fp8_moe_layer_shared_apart@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention, on one rank, with fp8 experts and "
        "the shared expert outside them: the residual stream hidden_states through "
        "post_attention_layernorm, then on it each token's topk experts by sigmoid score plus "
        "correction_bias, weighed by the renormalized unbiased scores times "
        "routed_scaling_factor, each expert down(silu(gate(x)) * up(x)), plus the shared expert "
        "(DeepseekV2MLP) at weight 1; every one of their linears is fp8 at a scale per 128 x 128 "
        "block and takes its input as the checkpoint's FP8 scheme quantizes it: in bf16, then "
        "each 1 x 128 group in fp8 at a scale of its amax / 448; out is hidden_states plus their "
        "output summed over the ranks of the kernel's peer memory in rank order, the stream "
        "after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size.", 128, 16384, multiple_of=128),
        "shared_inter_dim": axis(
            "The shared expert's intermediate size on this rank.",
            128,
            65536,
            multiple_of=128,
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "shared_gate_up_rows": derived_axis(
            "shared_gate_up_weight's rows: gate's, then up's.", "2 * shared_inter_dim"
        ),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
        "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
        "shared_gate_up_blocks": derived_axis(
            "128-blocks of shared_gate_up_weight's rows.", "shared_gate_up_rows // 128"
        ),
        "shared_inter_blocks": derived_axis(
            "128-blocks of the shared expert's intermediate size.",
            "shared_inter_dim // 128",
        ),
    },
    attrs=_ROUTING_ATTRS,
    constraints=("topk <= n_routed",),
    inputs={
        **_POST_NORM,
        **_ROUTING_INPUTS,
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_routed", "gate_up_rows", "model_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="w1's dequant scale per 128 x 128 block.",
            shape=("n_routed", "gate_up_blocks", "model_blocks"),
            dtype=_FLOAT32,
            distribution=_BLOCK_SCALE,
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_routed", "model_dim", "inter_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="inter_dim"),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="w2's dequant scale per 128 x 128 block.",
            shape=("n_routed", "model_blocks", "inter_blocks"),
            dtype=_FLOAT32,
            distribution=_BLOCK_SCALE,
            lifetime="weight",
        ),
        "shared_gate_up_weight": TensorSpec(
            description=("The shared expert's gate_up_proj on this rank: gate_proj's rows, then up_proj's."),
            shape=("shared_gate_up_rows", "model_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "shared_gate_up_scale": _block_scale("shared_gate_up_blocks", "model_blocks"),
        "shared_down_weight": TensorSpec(
            description="The shared expert's columns of down_proj on this rank.",
            shape=("model_dim", "shared_inter_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="shared_inter_dim"),
            lifetime="weight",
        ),
        "shared_down_scale": _block_scale("model_blocks", "shared_inter_blocks"),
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    outputs={"out": _layer_output(40.0)},
    # sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them, and its
    # fp8 linears the same way
    layout_variants=_shuffled("w1", "w2", "shared_gate_up_weight", "shared_down_weight"),
    compare=_COMPARE,
)

DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER = OpContract(
    contract_id="deepseek_v2_unquantized_moe_layer@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention, on one rank, with unquantized "
        "experts: the residual stream hidden_states through post_attention_layernorm, then on it "
        "each token's topk experts by sigmoid score plus correction_bias, weighed by the "
        "renormalized unbiased scores times routed_scaling_factor, and the shared expert in the "
        "last slot at weight 1, each expert down(silu(gate(x)) * up(x)) of its bf16 weights; out "
        "is hidden_states plus their output summed over the ranks of the kernel's peer memory in "
        "rank order, the stream after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size.", 32, 16384, multiple_of=32),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
    },
    attrs=_ROUTING_ATTRS,
    constraints=("topk <= n_routed",),
    inputs={
        **_POST_NORM,
        **_ROUTING_INPUTS,
        "w1": TensorSpec(
            description="Gate-up projection per expert slot: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert slot.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="inter_dim"),
            lifetime="weight",
        ),
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    outputs={"out": _layer_output(25.0)},
    # sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them
    layout_variants=_shuffled("w1", "w2"),
    compare=_COMPARE,
)

DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART = OpContract(
    contract_id="deepseek_v2_unquantized_moe_layer_shared_apart@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention, on one rank, with unquantized "
        "experts and the shared expert outside them: the residual stream hidden_states through "
        "post_attention_layernorm, then on it each token's topk experts by sigmoid score plus "
        "correction_bias, weighed by the renormalized unbiased scores times "
        "routed_scaling_factor, each expert down(silu(gate(x)) * up(x)) of its bf16 weights, "
        "plus the shared expert "
        "(DeepseekV2MLP) at weight 1; out is hidden_states plus their output summed over the "
        "ranks of the kernel's peer memory in rank order, the stream after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size.", 32, 16384, multiple_of=32),
        "shared_inter_dim": axis(
            "The shared expert's intermediate size on this rank.",
            32,
            65536,
            multiple_of=32,
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "shared_gate_up_rows": derived_axis(
            "shared_gate_up_weight's rows: gate's, then up's.", "2 * shared_inter_dim"
        ),
    },
    attrs=_ROUTING_ATTRS,
    constraints=("topk <= n_routed",),
    inputs={
        **_POST_NORM,
        **_ROUTING_INPUTS,
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_routed", "gate_up_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_routed", "model_dim", "inter_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="inter_dim"),
            lifetime="weight",
        ),
        "shared_gate_up_weight": TensorSpec(
            description=("The shared expert's gate_up_proj on this rank: gate_proj's rows, then up_proj's."),
            shape=("shared_gate_up_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "shared_down_weight": TensorSpec(
            description="The shared expert's columns of down_proj on this rank.",
            shape=("model_dim", "shared_inter_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="shared_inter_dim"),
            lifetime="weight",
        ),
        "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
    },
    outputs={"out": _layer_output(25.0)},
    # sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them
    layout_variants=_shuffled("w1", "w2"),
    compare=_COMPARE,
)

DEEPSEEK_V2_DENSE_LAYER = OpContract(
    contract_id="deepseek_v2_dense_layer@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention in a dense layer, on one rank, bf16 "
        "weights: the residual stream hidden_states through post_attention_layernorm, then "
        "DeepseekV2MLP on it: silu(gate) * up times down_weight^T, gate and up its products with "
        "gate_up_weight's two halves; out is hidden_states plus that output summed over the "
        "ranks of the kernel's peer memory in rank order, the stream after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "inter_dim": axis("This rank's intermediate size.", 128, 65536, multiple_of=128),
        "gate_up_rows": derived_axis("gate_up_weight's rows: gate's, then up's.", "2 * inter_dim"),
    },
    inputs={
        **_POST_NORM,
        "gate_up_weight": TensorSpec(
            description="This rank's rows of gate_up_proj: gate_proj's, then up_proj's.",
            shape=("gate_up_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "down_weight": TensorSpec(
            description="This rank's columns of down_proj.",
            shape=("model_dim", "inter_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="inter_dim"),
            lifetime="weight",
        ),
    },
    outputs={"out": _layer_output(25.0)},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

DEEPSEEK_V2_FP8_DENSE_LAYER = OpContract(
    contract_id="deepseek_v2_fp8_dense_layer@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer after its attention in a dense layer, on one rank, fp8 "
        "weights at a scale per 128 x 128 block: the residual stream hidden_states through "
        "post_attention_layernorm, then DeepseekV2MLP on it: silu(gate) * up times down_weight^T, "
        "gate and up its products with gate_up_weight's two halves, each of the two linears "
        "taking its input as the checkpoint's FP8 scheme quantizes it: in bf16, then each "
        "1 x 128 group in fp8 at a scale of its amax / 448; out is hidden_states plus that "
        "output summed over the ranks of the kernel's peer memory in rank order, the stream "
        "after the layer."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "inter_dim": axis("This rank's intermediate size.", 128, 65536, multiple_of=128),
        "gate_up_rows": derived_axis("gate_up_weight's rows: gate's, then up's.", "2 * inter_dim"),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "gate_up_blocks": derived_axis("128-blocks of gate_up_weight's rows.", "gate_up_rows // 128"),
        "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
    },
    inputs={
        **_POST_NORM,
        "gate_up_weight": TensorSpec(
            description="This rank's rows of gate_up_proj: gate_proj's, then up_proj's.",
            shape=("gate_up_rows", "model_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "gate_up_scale": _block_scale("gate_up_blocks", "model_blocks"),
        "down_weight": TensorSpec(
            description="This rank's columns of down_proj.",
            shape=("model_dim", "inter_dim"),
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="inter_dim"),
            lifetime="weight",
        ),
        "down_scale": _block_scale("model_blocks", "inter_blocks"),
    },
    outputs={"out": _layer_output(45.0)},
    # sglang holds fp8 linears for aiter as shuffle_weight(w, (16, 16)) leaves them
    layout_variants=_shuffled("gate_up_weight", "down_weight"),
    compare=_COMPARE,
)

# The residual stream the attention layer contracts start from, and its input_layernorm
_INPUT_NORM = {
    "hidden_states": TensorSpec(
        description="The residual stream the layer starts from.",
        shape=("n_tokens", "model_dim"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "input_norm_weight": TensorSpec(
        description="input_layernorm's weight.",
        shape=("model_dim",),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "input_norm_eps": ScalarSpec(
        description="input_layernorm's epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=_EPS,
    ),
}

# The attention's start, sglang's DeepseekV2AttentionMLA up to its attention, in every attention
# layer contract: from x, the residual stream through input_layernorm, qkv_a = x qkv_a_weight^T
# splits into q_a, the kv latent and k_pe; q_lora = RMSNorm(q_a), k_latent = RMSNorm(kv latent),
# q = q_lora q_b_weight^T (each head's q_nope, then its q_pe)
_QKV_AXES = {
    "q_lora_rank": axis("q_a's width.", 128, 16384, multiple_of=128),
    "qkv_a_rows": axis("qkv_a's rows.", 16, 32768, multiple_of=16),
    "rope_dim": derived_axis("k_pe's width.", "qkv_a_rows - q_lora_rank - kv_lora_rank"),
}
_QKV_BLOCKS = {
    "q_lora_blocks": derived_axis("128-blocks of q_a's width.", "q_lora_rank // 128"),
    "qkv_a_blocks": derived_axis("128-blocks of qkv_a's rows.", "ceil_div(qkv_a_rows, 128)"),
    "q_b_blocks": derived_axis("128-blocks of q's columns.", "ceil_div(q_b_rows, 128)"),
}
_QKV_INPUTS = {
    **_linear(
        "qkv_a",
        "fused_qkv_a_proj_with_mqa: q_a_proj's rows, then kv_a_proj_with_mqa's (the kv latent, then k_pe).",
        "qkv_a_rows",
        "model_dim",
        "qkv_a_blocks",
        "model_blocks",
    ),
    "q_a_norm_weight": TensorSpec(
        description="q_a_layernorm's weight.",
        shape=("q_lora_rank",),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "kv_a_norm_weight": TensorSpec(
        description="kv_a_layernorm's weight.",
        shape=("kv_lora_rank",),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "eps": ScalarSpec(
        description="Both RMSNorms' epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=_EPS,
    ),
    **_linear(
        "q_b",
        "This rank's rows of q_b_proj: each head's q_nope, then its q_pe.",
        "q_b_rows",
        "q_lora_rank",
        "q_b_blocks",
        "q_lora_blocks",
        same_as="qkv_a_weight",
    ),
}
_QKV_TEXT = (
    "x = RMSNorm(hidden_states) input_norm_weight; qkv_a = x qkv_a_weight^T splits into q_a, "
    "the kv latent and k_pe; q_lora = RMSNorm(q_a), k_latent = RMSNorm(kv latent), q = q_lora "
    "q_b_weight^T. An fp8 weight is dequantized per 128 x 128 block and takes its input as the "
    "checkpoint's FP8 scheme quantizes it: in bf16, as the model holds its activations, then "
    "each 1 x 128 group in fp8 at a scale of its amax / 448."
)

# The attention from q, k_latent and k_pe on, sparse (DSA) with kv_b_proj absorbed, as sglang's
# DeepseekV2AttentionMLA runs it, then o_proj, in every attention layer contract
_SPARSE_AXES = {
    "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
    "n_heads": axis("Heads on this rank.", 1, 128),
    "nope_dim": axis("A head's q_nope.", 16, 512, multiple_of=16),
    "kv_lora_rank": axis("The kv latent's width.", 128, 4096, multiple_of=128),
    "v_head_dim": axis("A head's output.", 16, 1024, multiple_of=16),
    "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
    "n_slots": axis("Rows of the KV cache.", 1, 2**31 - 1, is_var=True, any_size=True),
    "n_positions": axis("Rows of the RoPE tables.", 1, 2**31 - 1, is_var=True, any_size=True),
    "topk": axis("Top-k entries per token.", 1, 65536),
}
_SPARSE_DERIVED = {
    "kv_dim": derived_axis("A KV row.", "kv_lora_rank + rope_dim"),
    "o_cols": derived_axis("o_proj's input.", "n_heads * v_head_dim"),
    "half_rope": derived_axis("RoPE pairs.", "rope_dim // 2"),
    "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
    "o_blocks": derived_axis("128-blocks of o_proj's input.", "ceil_div(o_cols, 128)"),
}
_SPARSE_INPUTS = {
    "positions": TensorSpec(
        description="Each token's position.",
        shape=("n_tokens",),
        dtype=DTypeSpec(any_of=(DType.INT64,)),
        distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
    ),
    "out_cache_loc": TensorSpec(
        description="Each token's slot in the KV cache.",
        shape=("n_tokens",),
        dtype=DTypeSpec(any_of=(DType.INT64,)),
        distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
    ),
    "kv_cache": TensorSpec(
        description=(
            "This layer's KV cache: a row per slot, the latent then the RoPE part; the "
            "tokens' rows are stored into it."
        ),
        shape=("n_slots", "kv_dim"),
        dtype=_FP8,
        distribution=_FINITE,
        mutable=True,
        # the stored rows' fp8 rounding: relative, and absolute where RoPE's products cancel
        error_bounds=ToleranceBound(absolute_tolerance=2.0**-5, relative_tolerance=2.0**-3),
    ),
    "topk_indices": TensorSpec(
        description="The slots each token attends; -1: none.",
        shape=("n_tokens", "topk"),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
        distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
    ),
    "sm_scale": ScalarSpec(
        description="Multiplies every score.",
        dtype=DType.FLOAT32,
        interval=_POSITIVE_FLOAT32,
        default=_SM_SCALE,
    ),
    "cos_cache": TensorSpec(
        description="RoPE's cosines by position and pair.",
        shape=("n_positions", "half_rope"),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "sin_cache": TensorSpec(
        description="RoPE's sines by position and pair.",
        shape=("n_positions", "half_rope"),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "w_uk_t": TensorSpec(
        description="kv_b_proj's W_UK per head, transposed: (latent, q_nope).",
        shape=("n_heads", "kv_lora_rank", "nope_dim"),
        dtype=DTypeSpec(any_of=_WEIGHT_DTYPES),
        distribution=LecunNormalDistribution(fan_in="nope_dim"),
        lifetime="weight",
    ),
    "w_uv": TensorSpec(
        description="kv_b_proj's W_UV per head: (output, latent).",
        shape=("n_heads", "v_head_dim", "kv_lora_rank"),
        dtype=DTypeSpec(any_of=_WEIGHT_DTYPES, same_as_tensor="w_uk_t"),
        distribution=LecunNormalDistribution(fan_in="kv_lora_rank"),
        lifetime="weight",
    ),
    "w_scale": TensorSpec(
        description="fp8 w_uk_t's and w_uv's dequant scale.",
        shape=(1,),
        dtype=_FLOAT32,
        can_squeeze=True,
        present_when=_has("w_scale"),
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="1"),
        lifetime="weight",
    ),
    **_linear(
        "o_proj",
        "This rank's columns of o_proj.",
        "model_dim",
        "o_cols",
        "model_blocks",
        "o_blocks",
    ),
}
_SPARSE_TEXT = (
    "RoPE turns pair (x[2i], x[2i + 1]) by cos_cache and sin_cache at the token's position. A "
    "token's KV row is k_latent, then RoPE(k_pe), stored into kv_cache at its slot of "
    "out_cache_loc. Each head attends with q = (q_nope w_uk_t^T w_scale, RoPE(q_pe)) over the KV "
    "cache rows the token's top-k slots name (-1: none), the rows just stored among them: scores "
    "q . row * sm_scale, values the rows' latent part (0 when no row). The heads' outputs times "
    "w_uv w_scale, then o_proj (an fp8 one as the linears above). w_scale comes with "
    "fp8 w_uk_t and w_uv only; bf16 ones are as they are."
)
_Q_B_ROWS = derived_axis(
    "q's columns: the rank's heads times a head's q_nope and q_pe.",
    "n_heads * (nope_dim + rope_dim)",
)
_ATTENTION_OUTPUT = TensorSpec(
    description="The residual stream after the attention.",
    shape=("n_tokens", "model_dim"),
    dtype=_BF16,
    error_bounds=SqnrBound(min_sqnr_db=20.0),
)
_OUTPUT_TEXT = (
    "output is hidden_states plus o_proj's output summed over the ranks of the kernel's peer "
    "memory, in rank order: the stream after the attention."
)
# (scale, its weight, the attr's first value): the linears' scales absent first (GLM-5.2's bf16
# linears), so no name passes 255 bytes
_ATTENTION_SCALES = (
    ("qkv_a_scale", "qkv_a_weight", False),
    ("q_b_scale", "q_b_weight", False),
    ("o_proj_scale", "o_proj_weight", False),
    ("w_scale", "w_uk_t", True),
)
_ATTENTION_ATTRS = {_has(s): _scale_attr(s, w, first) for s, w, first in _ATTENTION_SCALES}
_ATTENTION_SCALE_CONSTRAINTS = tuple(_scale_constraint(s, w) for s, w, _ in _ATTENTION_SCALES)

DEEPSEEK_V2_ATTENTION = OpContract(
    contract_id="deepseek_v2_attention@1.0",
    description=(
        "sglang's DeepseekV2DecoderLayer up to its MLP on one rank, in the layers that attend a "
        f"carried top-k (no indexer), from the residual stream hidden_states: {_QKV_TEXT} "
        f"{_SPARSE_TEXT} The top-k slots are topk_indices. {_OUTPUT_TEXT}"
    ),
    op_type="deepseek_v2",
    axes={
        **_SPARSE_AXES,
        **_QKV_AXES,
        "q_b_rows": _Q_B_ROWS,
        **_SPARSE_DERIVED,
        **_QKV_BLOCKS,
    },
    attrs=_ATTENTION_ATTRS,
    constraints=(
        "qkv_a_rows > q_lora_rank + kv_lora_rank",
        # distinct slots, and distinct top-k entries with -1 among them
        "n_tokens <= n_slots",
        "topk <= n_slots + 1",
        *_ATTENTION_SCALE_CONSTRAINTS,
    ),
    inputs={**_INPUT_NORM, **_QKV_INPUTS, **_SPARSE_INPUTS},
    outputs={"output": _ATTENTION_OUTPUT},
    layout_variants=_shuffled("qkv_a_weight", "q_b_weight", "o_proj_weight"),
    compare=_COMPARE,
)

# The DSA indexer, as sglang's Indexer runs it on its paged index-K cache, in the attention layer
# contracts with it: its weights and RoPE tables (each weight in the shape of the axes below)
_INDEXER_INPUTS = {
    "wq_b": TensorSpec(
        description="wq_b: each head's rows in turn.",
        shape=("index_cols", "q_lora_rank"),
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in="q_lora_rank"),
        lifetime="weight",
    ),
    "wk": TensorSpec(
        description="wk.",
        shape=("ix_dim", "model_dim"),
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "weights_proj": TensorSpec(
        description="weights_proj: a row per head.",
        shape=("ix_heads", "model_dim"),
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "k_norm_weight": TensorSpec(
        description="k_norm's weight.",
        shape=("ix_dim",),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "k_norm_bias": TensorSpec(
        description="k_norm's bias.",
        shape=("ix_dim",),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "k_norm_eps": ScalarSpec(
        description="k_norm's epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=1e-6,
    ),
    "index_cos_cache": TensorSpec(
        description="RoPE's cosines by position and pair.",
        shape=("n_positions", "half_rope"),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "index_sin_cache": TensorSpec(
        description="RoPE's sines by position and pair.",
        shape=("n_positions", "half_rope"),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
}
_INDEXER_TEXT = (
    "The indexer's q is each head of q_lora wq_b^T and its k is LayerNorm(x wk^T), both turned "
    "by RoPE on their first 2 half_rope values (by index_cos_cache and index_sin_cache), then the "
    "Hadamard transform over ix_dim times ix_dim^-0.5 (fast_hadamard_transform's order); the "
    "heads' weights are x weights_proj^T * ix_heads^-0.5 * ix_dim^-0.5. Each token's k is "
    "stored into index_k_cache at its slot of out_cache_loc (page slot // 64, key slot % 64) in "
    "fp8 at a scale of 2^ceil(log2(max(amax, 1e-4) / 448)) of its values. The scores: for each "
    "of a token's first seq_lens[m] positions t, sum_h weights[h] relu(<q[h], k(t)>), k(t) the "
    "key at slot page_table[m, t // 64] * 64 + t % 64: one of this call's k, else the cache's "
    "values times their scale. topk_indices are their topk highest (all of them in position "
    "order when there are no more), each position t as its slot, -1 past them: the token's "
    "top-k slots. Ties and a full row's order are the kernel's choice."
)

# deepseek_v2_attention_indexer's description, axes, constraints, inputs and outputs, which
# deepseek_v2_fp8_attention_indexer shares
_ATTENTION_INDEXER_TEXT = (
    "sglang's DeepseekV2DecoderLayer up to its MLP on one rank, in the layers that run the DSA "
    f"indexer, from the residual stream hidden_states: {_QKV_TEXT} {_INDEXER_TEXT} {_SPARSE_TEXT} "
    f"{_OUTPUT_TEXT}"
)
_ATTENTION_INDEXER_AXES = {
    **_SPARSE_AXES,
    **_QKV_AXES,
    "ix_heads": axis("Indexer heads.", 1, 128),
    "ix_dim": axis("A key and a head: the pages' 128.", 128, 128),
    "n_pages": axis("Cache pages.", 1, 2**31 - 1, is_var=True, any_size=True),
    "table_width": axis("Page table columns.", 1, 2**25, is_var=True, any_size=True),
    "q_b_rows": _Q_B_ROWS,
    **_SPARSE_DERIVED,
    **_QKV_BLOCKS,
    "index_cols": derived_axis(
        "wq_b's rows: the indexer heads times a head.",
        "ix_heads * ix_dim",
    ),
    "page_bytes": derived_axis(
        "A page: 64 keys of ix_dim fp8 values, then their fp32 scales.",
        "64 * (ix_dim + 4)",
    ),
}
_ATTENTION_INDEXER_CONSTRAINTS = (
    "qkv_a_rows > q_lora_rank + kv_lora_rank",
    "2 * half_rope <= ix_dim",
    # a slot names a KV row and a key of the index-K cache, distinct ones the tokens'
    "n_slots == 64 * n_pages",
    "n_tokens <= n_slots",
    "table_width <= n_pages",
)
_ATTENTION_INDEXER_INPUTS = {
    **_INPUT_NORM,
    **_QKV_INPUTS,
    **_INDEXER_INPUTS,
    **{n: _SPARSE_INPUTS[n] for n in ("positions", "out_cache_loc")},
    "index_k_cache": TensorSpec(
        description=(
            "This layer's index-K cache: pages of 64 keys' fp8 values in 16 x 16 tiles (16 "
            "keys by 16 values), then their fp32 scales; the tokens' keys are stored into it."
        ),
        shape=("n_pages", "page_bytes"),
        dtype=_FP8,
        distribution=DsaIndexKCacheDistribution(),
        mutable=True,
        # in fp8 units: the transform spreads the kernel's bf16 roundings before it to a couple
        # of units; where a key's amax sits at a power of two, its scale may be a step from the
        # reference's and its values a factor of 2, each then one more fp8 rounding (2^-4) off
        error_bounds=ToleranceBound(absolute_tolerance=2.0, relative_tolerance=1.0 + 2.0**-4),
    ),
    "seq_lens": TensorSpec(
        description=(
            "Each token's positions, its own included; at most 2^20, the DSA models' longest context."
        ),
        shape=("n_tokens",),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
        distribution=IntRangeDistribution(lo="1", hi="min(64 * table_width, 1048576)"),
    ),
    "page_table": TensorSpec(
        description="Each token's pages in position order, distinct within a row.",
        shape=("n_tokens", "table_width"),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
        distribution=IntRangeDistribution(lo="0", hi="n_pages - 1", distinct=True),
    ),
    **{n: s for n, s in _SPARSE_INPUTS.items() if n not in ("positions", "out_cache_loc", "topk_indices")},
}
_ATTENTION_INDEXER_OUTPUTS = {
    "output": _ATTENTION_OUTPUT,
    "topk_indices": TensorSpec(
        description="The slots each token attends; -1: none.",
        shape=("n_tokens", "topk"),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
    ),
}

DEEPSEEK_V2_ATTENTION_INDEXER = OpContract(
    contract_id="deepseek_v2_attention_indexer@1.0",
    description=_ATTENTION_INDEXER_TEXT,
    op_type="deepseek_v2",
    axes=_ATTENTION_INDEXER_AXES,
    attrs=_ATTENTION_ATTRS,
    constraints=(*_ATTENTION_INDEXER_CONSTRAINTS, *_ATTENTION_SCALE_CONSTRAINTS),
    inputs=_ATTENTION_INDEXER_INPUTS,
    outputs=_ATTENTION_INDEXER_OUTPUTS,
    layout_variants=_shuffled("qkv_a_weight", "q_b_weight", "o_proj_weight"),
    compare=_COMPARE,
)

# deepseek_v2_attention_indexer with every linear fp8, as GLM-5.3-FP8 holds them, the indexer's
# wq_b and wk too: a contract apart, as presence attrs of their scales would put the longest name
# of deepseek_v2_attention_indexer past 255 bytes. Each scale is passed; wq_b's and wk's come
# after their weights.
_FP8_WEIGHTS = (
    "qkv_a_weight",
    "q_b_weight",
    "wq_b",
    "wk",
    "w_uk_t",
    "w_uv",
    "o_proj_weight",
)
_INDEXER_SCALES = {
    "wq_b": ("wq_b_scale", _block_scale("wq_b_blocks", "q_lora_blocks")),
    "wk": ("wk_scale", _block_scale("wk_blocks", "model_blocks")),
}


def _fp8_inputs(inputs: dict) -> dict:
    """``inputs`` with every weight of _FP8_WEIGHTS fp8, each scale passed, and the indexer's
    scales after its weights."""
    fp8 = {}
    for name, spec in inputs.items():
        if name in _FP8_WEIGHTS:
            spec = spec.model_copy(update={"dtype": _FP8})
        elif isinstance(spec, TensorSpec) and spec.present_when is not None:
            spec = spec.model_copy(update={"present_when": None})
        fp8[name] = spec
        if name in _INDEXER_SCALES:
            scale, scale_spec = _INDEXER_SCALES[name]
            fp8[scale] = scale_spec
    return fp8


DEEPSEEK_V2_FP8_ATTENTION_INDEXER = OpContract(
    contract_id="deepseek_v2_fp8_attention_indexer@1.0",
    description=(
        f"{_ATTENTION_INDEXER_TEXT} Every linear is fp8: qkv_a_weight, q_b_weight, wq_b, wk and "
        "o_proj_weight with their scales per 128 x 128 block, wq_b and wk as the others, and "
        "w_uk_t and w_uv with w_scale."
    ),
    op_type="deepseek_v2",
    axes={
        **_ATTENTION_INDEXER_AXES,
        "wq_b_blocks": derived_axis("128-blocks of wq_b's rows.", "ceil_div(index_cols, 128)"),
        "wk_blocks": derived_axis("128-blocks of wk's rows.", "ceil_div(ix_dim, 128)"),
    },
    constraints=_ATTENTION_INDEXER_CONSTRAINTS,
    inputs=_fp8_inputs(_ATTENTION_INDEXER_INPUTS),
    outputs=_ATTENTION_INDEXER_OUTPUTS,
    layout_variants=_shuffled("qkv_a_weight", "q_b_weight", "o_proj_weight", "wq_b", "wk"),
    compare=_COMPARE,
)

DEEPSEEK_V2_EH_PROJ = OpContract(
    contract_id="deepseek_v2_eh_proj@1.0",
    description=(
        "sglang's DeepseekModelNextN (the MTP draft layer) from its inputs to its decoder's: "
        "eh_proj (bf16, every rank holding all of it) of cat(RMSNorm(embeds) enorm_weight, "
        "RMSNorm(hidden) hnorm_weight). The ranks of the kernel's peer memory each take an "
        "equal share of eh_proj's columns, and their products are summed in rank order."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "eh_cols": derived_axis("eh_proj's input: both normed halves.", "2 * model_dim"),
    },
    inputs={
        "embeds": TensorSpec(
            description="The draft tokens' embeddings.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "hidden": TensorSpec(
            description="The target model's hidden states the drafts continue.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "enorm_weight": TensorSpec(
            description="enorm's weight.",
            shape=("model_dim",),
            dtype=_BF16,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "hnorm_weight": TensorSpec(
            description="hnorm's weight.",
            shape=("model_dim",),
            dtype=_BF16,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "eps": ScalarSpec(
            description="Both norms' epsilon.",
            dtype=DType.FLOAT32,
            interval=(0.0, 1.0),
            default=_EPS,
        ),
        "weight": TensorSpec(
            description="eh_proj's weight, row-major as sglang holds it.",
            shape=("model_dim", "eh_cols"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="eh_cols"),
            lifetime="weight",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="The decoder's input.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            error_bounds=SqnrBound(min_sqnr_db=25.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

CONTRACTS = (
    DEEPSEEK_V2_MOE,
    DEEPSEEK_V2_MOE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART,
    DEEPSEEK_V2_DENSE_LAYER,
    DEEPSEEK_V2_FP8_DENSE_LAYER,
    DEEPSEEK_V2_ATTENTION,
    DEEPSEEK_V2_ATTENTION_INDEXER,
    DEEPSEEK_V2_FP8_ATTENTION_INDEXER,
    DEEPSEEK_V2_EH_PROJ,
)
