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
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.sglang.models import moe_blocks

__all__ = [
    "DEEPSEEK_V2_FP8_MOE",
    "DEEPSEEK_V2_MXFP4_MOE",
    "declare",
    "deepseek_v2_fp8_moe",
    "deepseek_v2_mxfp4_moe",
    "family",
]


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

_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")


def _fp8_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    return tuple(
        LayoutFact(location="tensor_attr", name=f"{w}.is_shuffled", expected_value=shuffled)
        for w in ("w1", "w2")
    )


DEEPSEEK_V2_FP8_MOE = OpContract(
    contract_id="deepseek_v2_fp8_moe@1.0",
    description=(
        "sglang's DeepseekV2MoE on one rank, with fp8 experts at a scale per 128 x 128 "
        "block and the shared expert in their last slot: on hidden_states, the router's "
        "logits in hidden_states' dtype (as sglang's MoEGate returns them on ROCm) and "
        "their sigmoid scores; the scores plus correction_bias rank the experts, which "
        "fall into num_expert_group equal groups, each scored by its two best biased "
        "scores summed; each token's topk experts are its best biased scores within its "
        "topk_group best groups (the lower index first among equals), weighed by the "
        "renormalized unbiased scores times routed_scaling_factor, and the shared expert "
        "at weight 1; each expert down(silu(gate(x)) * up(x)) of its fp8 weights, each of "
        "its two linears taking its input as the checkpoint's FP8 scheme quantizes it: in "
        "bf16, then each 1 x 128 group in fp8 at a scale of its amax / 448; out is their "
        "output on this rank, which sglang sums over the ranks."
    ),
    op_type="deepseek_v2",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
        "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
    },
    attrs={
        "topk": attr("Routed experts per token.", 8, param_name="topk"),
        # DeepSeek-V3/R1's 8 groups, then GLM-5.2's one: its experts ungrouped
        "num_expert_group": attr("Expert groups.", 8, 1, param_name="num_expert_group"),
        "topk_group": attr("Groups each token's experts come from.", 4, 1, param_name="topk_group"),
        "renormalize": attr(
            "Whether the routed weights are divided by their sum.",
            True,
            param_name="renormalize",
        ),
        "scoring_func": attr("Router score.", "sigmoid", param_name="scoring_func"),
    },
    constraints=(
        "n_routed % num_expert_group == 0",
        # a group's score sums its two best experts
        "n_routed // num_expert_group >= 2",
        "topk_group <= num_expert_group",
        "topk <= topk_group * (n_routed // num_expert_group)",
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
            # bf16, as sglang's aiter top-k reads it for fp8 checkpoints: DeepSeek-R1 holds it in
            # bf16, GLM-5.2 in fp32, which top-k reads through a bf16 copy (the route passes that)
            # TODO: fp32 for an fp32 bias (GLM-5.x) once the sglang AKLR targets has sglang PR
            # #44797, whose aiter top-k reads an fp32 bias as it is; 0.5.22 and 0.5.23 do not
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            # centred on 0 and narrow beside the scores' spread: it moves most tokens' picks,
            # yet keeps the routes spread over the experts, as a balanced model's are
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
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
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies the routed weights only: DeepSeek-V3's and R1's.",
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
            # implementations round the second linear's input to fp8 from values they round
            # at other points than the reference does, which flips some fp8 codes: a CPU
            # emulation of the kernels' rounding comes to 35 to 36 dB on the bench's inputs
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    # row-major, or as sglang holds the experts for aiter: shuffle_weight's 16 x 16 tiles,
    # the scales as the checkpoint stores them
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_fp8_facts(False)),
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(("w1", "w2"), Preshuffle(tile=(16, 16))),
                detect_facts=_fp8_facts(True),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("sglang")
family.define(DEEPSEEK_V2_MXFP4_MOE)
family.define(DEEPSEEK_V2_FP8_MOE)
family.build()


@family.op
def deepseek_v2_mxfp4_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@family.op
def deepseek_v2_fp8_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


def _moe_forward_normal(
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    input_ids=None,
    input_ids_global=None,
    skip_shared_experts=False,
    use_vision_topk=False,
    return_moe_output=False,
    leave_output_sum=False,
):
    """The MoE's router, top-k and experts, the shared one last, as one op call; then
    forward_normal's all-reduce unless a later step sums the ranks."""
    if _forward_normal_falls_back(
        self, skip_shared_experts, use_vision_topk, return_moe_output or leave_output_sum
    ):
        return FallbackReason.OUT_OF_CONTRACT
    op, arguments = _moe_call(self, hidden_states)
    out = op(**arguments)
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)


# the experts' formats the ops take: Quark's MXFP4, fp8 at a scale per 128 x 128 block
_EXPERT_FORMATS = ("QuarkW4A4MXFp4MoE", "Fp8MoEMethod")


def _forward_normal_falls_back(mlp, skip_shared_experts, use_vision_topk, return_moe_output) -> bool:
    # a rank's share of the experts (expert parallelism) or the routed ones alone (the shared
    # expert apart) are another shape than the op's w1, which holds the shared one last
    return (
        skip_shared_experts
        or use_vision_topk
        or return_moe_output
        or _masks_padded_rows()
        or moe_blocks.runtime_falls_back()
        or moe_blocks.experts_fall_back(mlp.experts, _EXPERT_FORMATS)
        or mlp._fuse_shared_experts_inside_sbo
        or mlp.is_hash
        or mlp.gate.e_score_correction_bias_vl is not None
    )


def _masks_padded_rows() -> bool:
    """Whether sglang's top-k masks padded rows here: the bound it reads from the forward context
    (resolve_routing_inputs), forward_normal's num_token_non_padded until 0.5.23."""
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context

    if not has_forward_context():
        return False
    get_bound = get_forward_context().moe_num_token_non_padded
    return get_bound is not None and get_bound() is not None


def _moe_call(mlp, hidden_states):
    """The op for the experts' format, and its arguments: the module's weights and input."""
    experts, routing = mlp.experts, mlp.topk.topk_config
    grouped = routing.use_grouped_topk
    arguments = {
        "hidden_states": hidden_states,
        "router_weight": mlp.gate.weight,
        "correction_bias": routing.correction_bias,
        "w1": experts.w13_weight,
        "w2": experts.w2_weight,
        "topk": routing.top_k - routing.num_fused_shared_experts,
        "num_expert_group": routing.num_expert_group if grouped else 1,
        "topk_group": routing.topk_group if grouped else 1,
        "renormalize": routing.renormalize,
        "scoring_func": routing.scoring_func,
        "routed_scaling_factor": mlp.routed_scaling_factor,
    }
    if moe_blocks.experts_format(experts) == "Fp8MoEMethod":
        return deepseek_v2_fp8_moe, {
            **arguments,
            "correction_bias": _topk_bias(routing, hidden_states.dtype),
            "w1_scale": experts.w13_weight_scale_inv,
            "w2_scale": experts.w2_weight_scale_inv,
        }
    return deepseek_v2_mxfp4_moe, {
        **arguments,
        "w1_scale": experts.w13_weight_scale,
        "w2_scale": experts.w2_weight_scale,
    }


def _topk_bias(routing, logits_dtype):
    """``routing``'s correction bias as sglang's select_experts hands it to the top-k: in the
    router logits' dtype to aiter's grouped top-k (the copy TopKConfig keeps), else as held.
    GLM-5.2 holds its bias in fp32, which aiter's top-k reads in bf16."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.topk import _use_aiter

    if _use_aiter and routing.use_grouped_topk and not envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.get():
        return routing.correction_bias_for_dtype(logits_dtype)
    return routing.correction_bias


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal",
        versions=("0.5.23",),
        replace=_moe_forward_normal,
    )
