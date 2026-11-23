# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's Qwen3MoeSparseMoeBlock.forward_normal with fp8 experts (aiter's path): the
router, the top-k and the experts as one op. Qwen3-MoE has no shared expert.

A model is a scenario of the contract: Qwen3-Coder-480B-A35B FP8 at TP4 is model_dim 6144,
n_experts 160, inter_dim 640 (2560 split over the 4 ranks), topk 8."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
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
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.sglang.models import moe_blocks

__all__ = ["QWEN3_MOE_FP8_MOE", "declare", "family", "qwen3_moe_fp8_moe"]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")

# sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them (and marks
# them is_shuffled), the scales as the checkpoint stores them
_EXPERT_LAYOUTS = LayoutVariantsSpec(
    by_name={
        name: LayoutVariant(
            layouts=(dict.fromkeys(("w1", "w2"), Preshuffle(tile=(16, 16))) if shuffled else {}),
            detect_facts=tuple(
                LayoutFact(
                    location="tensor_attr",
                    name=f"{w}.is_shuffled",
                    expected_value=shuffled,
                )
                for w in ("w1", "w2")
            ),
        )
        for name, shuffled in (("row_major", False), ("shuffled_16x16", True))
    }
)

QWEN3_MOE_FP8_MOE = OpContract(
    contract_id="qwen3_moe_fp8_moe@1.0",
    description=(
        "sglang's Qwen3MoeSparseMoeBlock.forward_normal on one rank of its TP group with fp8 "
        "experts, every expert on the rank: the logits hidden_states router_weight^T, rounded "
        "to bf16, pick each token's topk experts by descending logit, a tie going to the "
        "lower expert id, weighed by the softmax of their logits (the softmax over every "
        "expert, renormalized over the chosen ones); each expert is down(silu(gate(x)) * "
        "up(x)) of fp8 weights at a scale per 128 x 128 block, each of its two linears taking "
        "its input as the checkpoint's FP8 scheme quantizes it: in bf16, then each 1 x 128 "
        "group in fp8 at a scale of its amax / 448; out is this rank's share of the block's "
        "output, which the TP group sums after the call."
    ),
    op_type="qwen3_moe",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_experts": axis("Experts: the router's rows, every one held on this rank.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
        "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
    },
    attrs={"topk": attr("Experts per token.", 8, param_name="topk")},
    constraints=("topk <= n_experts",),
    inputs={
        "hidden_states": TensorSpec(
            description="The tokens' hidden states, the MoE block's input.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "router_weight": TensorSpec(
            description="The router (the block's gate) projection.",
            shape=("n_experts", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert on this rank: gate rows, then up rows.",
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
            description="Down projection per expert on this rank.",
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
    },
    # Implementations quantize where the reference does but compute each 1 x 128 scale as
    # amax * (1 / 448) (aiter's per-group quant then takes x * (1 / scale), most kernels
    # x / scale), so the fp8 codes of values on an exact rounding tie differ, which
    # dominates their error: emulated in float64, sglang's path and the kernels come to 34
    # to 38 dB at Qwen3-Coder's shape, 30 dB in the worst of 50 draws of the small test
    # Point (median 46).
    outputs={
        "out": TensorSpec(
            description="This rank's share of the MoE block's output.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    layout_variants=_EXPERT_LAYOUTS,
    # atomics order some kernels' routing queues
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("sglang")
family.define(QWEN3_MOE_FP8_MOE)
family.build()


@family.op
def qwen3_moe_fp8_moe(
    hidden_states,
    router_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=8,
): ...


def _moe_forward_normal(self, hidden_states):
    """The block's gate, top-k and experts on this rank as one op call; sglang's layer
    boundary sums the ranks' shares after it, as it does sglang's own."""
    if _masks_padded_rows() or moe_blocks.runtime_falls_back() or _block_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    experts = self.experts
    return qwen3_moe_fp8_moe(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale_inv,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale_inv,
        topk=self.topk.topk_config.top_k,
    )


def _masks_padded_rows() -> bool:
    """Whether sglang's top-k masks padded rows here: since 0.5.23 it takes the bound from the
    forward context (resolve_routing_inputs), and the op routes every row."""
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context

    if not has_forward_context():
        return False
    get_bound = get_forward_context().moe_num_token_non_padded
    return get_bound is not None and get_bound() is not None


def _block_falls_back(block) -> bool:
    from sglang.srt.runtime_context import get_parallel

    gate = block.gate
    return (
        # the op indexes w1 by the router's ids over every expert, not this rank's
        get_parallel().moe_ep_size != 1
        # a LoRA wrapper, or a gate the checkpoint quantizes (ModelSlim)
        or type(gate).__name__ != "ReplicatedLinear"
        or type(gate.quant_method).__name__ != "UnquantizedLinearMethod"
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("Fp8MoEMethod",))
    )


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal",
        versions=("0.5.22", "0.5.23"),
        replace=_moe_forward_normal,
    )
