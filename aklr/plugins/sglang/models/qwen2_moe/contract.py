# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's Qwen2MoeSparseMoeBlock with its shared expert fused into the experts (aiter's
path for Qwen3.5 and Qwen3-Next): the router, the fp8 experts, and both at once.

A model is a scenario of each contract: Qwen3.5-397B-A17B at TP4 is model_dim 4096,
n_routed 512, inter_dim 256; Qwen3-Next-80B-A3B at TP4 is model_dim 2048, n_routed 512,
inter_dim 128."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
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

__all__ = [
    "CONTRACTS",
    "QWEN2_MOE_FP8_EXPERTS",
    "QWEN2_MOE_FP8_ROUTER_EXPERTS",
    "QWEN2_MOE_ROUTER",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
_COMPARE = CompareSpec(determinism=DeterminismLevel.BOUNDED)

_TOKENS = {
    "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
    "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
}
_ROUTED = axis("Routed experts: the router's rows.", 1, 1024)
_TOPK = axis("Routed experts per token.", 1, 64)
# the experts as sglang's FusedMoE holds them on one rank of its TP group: every expert,
# its intermediate size split over the ranks, the shared expert in the slot after the
# routed ones
_EXPERT_AXES = {
    "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
    "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
    "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
    "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
    "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
}
_SLOTS = "Expert slots: the routed experts, then the shared one."

_HIDDEN_STATES = TensorSpec(
    description="The tokens' hidden states, the MoE block's input.",
    shape=("n_tokens", "model_dim"),
    dtype=_BF16,
    distribution=FiniteDistribution(),
)
_ROUTER_WEIGHT = TensorSpec(
    description="The router (the block's gate) projection.",
    shape=("n_routed", "model_dim"),
    dtype=_BF16,
    distribution=LecunNormalDistribution(fan_in="model_dim"),
    lifetime="weight",
)
_SHARED_GATE_WEIGHT = TensorSpec(
    description=(
        "shared_expert_gate's weight: the shared expert weighs sigmoid(hidden_states shared_gate_weight^T)."
    ),
    shape=(1, "model_dim"),
    dtype=_BF16,
    distribution=LecunNormalDistribution(fan_in="model_dim"),
    lifetime="weight",
)
_EXPERTS = {
    "w1": TensorSpec(
        description=("Gate-up projection per expert slot on this rank: gate rows, then up rows."),
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
        description="Down projection per expert slot on this rank.",
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

_ROUTING = (
    "the logits hidden_states router_weight^T, rounded to bf16, pick each token's topk "
    "experts by descending logit, a tie going to the lower expert id, weighed by the "
    "softmax of their logits (the softmax over every expert, renormalized over the chosen "
    "ones)"
)
_SHARED = "and the shared expert by sigmoid(hidden_states shared_gate_weight^T)"
_EXPERT = (
    "each expert is down(silu(gate(x)) * up(x)) of fp8 weights at a scale per 128 x 128 "
    "block, each of its two linears taking its input as the checkpoint's FP8 scheme "
    "quantizes it: in bf16, then each 1 x 128 group in fp8 at a scale of its amax / 448; "
    "out is this rank's share of the block's output, which the TP group sums after the "
    "call"
)


# Implementations quantize the second linear's input from bf16 values they round at other
# points than the reference does, which flips some fp8 codes: about 33 to 38 dB on the
# bench's inputs. sglang's own gfx950 small-M experts keep bf16 activations, about 27 dB.
def _moe_output(min_sqnr_db: float) -> TensorSpec:
    return TensorSpec(
        description="This rank's share of the MoE block's output.",
        shape=("n_tokens", "model_dim"),
        dtype=DTypeSpec(same_as_tensor="hidden_states"),
        error_bounds=SqnrBound(min_sqnr_db=min_sqnr_db),
    )


QWEN2_MOE_ROUTER = OpContract(
    contract_id="qwen2_moe_router@1.0",
    description=(f"sglang's Qwen2MoeSparseMoeBlock router, its gate and TopK with renormalize: {_ROUTING}."),
    op_type="qwen2_moe",
    axes={**_TOKENS, "n_routed": _ROUTED, "topk": _TOPK},
    constraints=("topk <= n_routed",),
    inputs={"hidden_states": _HIDDEN_STATES, "router_weight": _ROUTER_WEIGHT},
    outputs={
        "topk_weights": TensorSpec(
            description="The chosen experts' weights, in topk_ids' order.",
            shape=("n_tokens", "topk"),
            dtype=_FLOAT32,
            # a logit summed in float32 may round to the next bf16 (2**-4 below 8),
            # which moves the weights by up to that
            error_bounds=ToleranceBound(absolute_tolerance=1e-6, relative_tolerance=2**-4),
        ),
        "topk_ids": TensorSpec(
            description="The chosen experts, by descending logit.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

QWEN2_MOE_FP8_EXPERTS = OpContract(
    contract_id="qwen2_moe_fp8_experts@1.0",
    description=(
        "sglang's Qwen2MoeSparseMoeBlock experts on one rank of its TP group, the shared "
        "expert fused in after the routed ones: each token's topk_ids experts weighed by "
        f"topk_weights, {_SHARED}; {_EXPERT}."
    ),
    op_type="qwen2_moe",
    axes={
        **_TOKENS,
        "n_experts": axis(_SLOTS, 2, 1025),
        "n_routed": derived_axis("Routed experts.", "n_experts - 1"),
        **_EXPERT_AXES,
        "topk": _TOPK,
    },
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": _HIDDEN_STATES,
        "topk_weights": TensorSpec(
            description="The routed experts' weights.",
            shape=("n_tokens", "topk"),
            dtype=_FLOAT32,
            # a renormalized top-k softmax's
            distribution=LogUniformDistribution(log2_lo="-6", log2_hi="-1"),
        ),
        "topk_ids": TensorSpec(
            description="Each token's routed experts, distinct within a row.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_routed - 1", distinct=True),
        ),
        "shared_gate_weight": _SHARED_GATE_WEIGHT,
        **_EXPERTS,
    },
    outputs={"out": _moe_output(25.0)},
    layout_variants=_EXPERT_LAYOUTS,
    compare=_COMPARE,
)

QWEN2_MOE_FP8_ROUTER_EXPERTS = OpContract(
    contract_id="qwen2_moe_fp8_router_experts@1.0",
    description=(
        "sglang's Qwen2MoeSparseMoeBlock._forward_router_experts on one rank of its TP "
        "group, the shared expert fused into the experts after the routed ones: "
        f"{_ROUTING}, {_SHARED}; {_EXPERT}."
    ),
    op_type="qwen2_moe",
    axes={
        **_TOKENS,
        "n_routed": _ROUTED,
        "n_experts": derived_axis(_SLOTS, "n_routed + 1"),
        **_EXPERT_AXES,
    },
    attrs={"topk": attr("Routed experts per token.", 10, param_name="topk")},
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": _HIDDEN_STATES,
        "router_weight": _ROUTER_WEIGHT,
        "shared_gate_weight": _SHARED_GATE_WEIGHT,
        **_EXPERTS,
    },
    outputs={"out": _moe_output(25.0)},
    layout_variants=_EXPERT_LAYOUTS,
    compare=_COMPARE,
)

CONTRACTS = (QWEN2_MOE_ROUTER, QWEN2_MOE_FP8_EXPERTS, QWEN2_MOE_FP8_ROUTER_EXPERTS)
