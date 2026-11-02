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

__all__ = ["CONTRACTS", "MINIMAX_M2_FP8_MOE", "MINIMAX_M2_MXFP4_MOE"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
# atomics order the routing queues and the combine of some kernels
_COMPARE = CompareSpec(determinism=DeterminismLevel.BOUNDED)

_ROUTING = (
    "each token's topk experts of the n_routed by the sigmoid of its fp32 router logits "
    "plus correction_bias, weighed by their unbiased scores renormalized to sum 1"
)
_RANK = (
    "out is the weighed sum over the experts this rank holds, the n_experts from "
    "expert_offset on (all n_routed without expert parallelism), the routes to the others "
    "adding nothing: this rank's share, not yet summed over the ranks"
)


def _axes(blocks: dict) -> dict:
    """The axes both contracts share, then the scale ``blocks`` of their format."""
    return {
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "n_experts": axis("Experts this rank holds: w1's and w2's.", 1, 1024),
        "inter_dim": axis(
            "Expert intermediate size on this rank, as stored (padding included).",
            128,
            16384,
            multiple_of=128,
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        **blocks,
    }


_ATTRS = {"topk": attr("Routed experts per token.", 8, param_name="topk")}
_CONSTRAINTS = ("topk <= n_routed", "n_experts <= n_routed")
_ROUTING_INPUTS = {
    "hidden_states": TensorSpec(
        description="The MoE's input: each token's hidden state.",
        shape=("n_tokens", "model_dim"),
        dtype=_BF16,
        distribution=FiniteDistribution(),
    ),
    "router_weight": TensorSpec(
        description="The gate: one fp32 row per routed expert.",
        shape=("n_routed", "model_dim"),
        dtype=_FLOAT32,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "correction_bias": TensorSpec(
        description="Added to the scores to pick experts; never weighs them.",
        shape=("n_routed",),
        dtype=_FLOAT32,
        # about 8, a spread like GLM-5's, as deepseek_v2_mxfp4_moe draws it
        distribution=LogUniformDistribution(log2_lo="2.99", log2_hi="3.01"),
        lifetime="weight",
    ),
}
_EXPERT_OFFSET = ScalarSpec(
    description=(
        "The router row of w1's first expert: this rank holds experts expert_offset to "
        "expert_offset + n_experts - 1, with expert_offset + n_experts <= n_routed (sglang "
        "passes moe_ep_rank * n_experts, one value per process: 0 without expert "
        "parallelism)."
    ),
    dtype=DType.INT32,
    interval=(0, 1023),
    default=0,
)


def _output(min_sqnr_db: float) -> TensorSpec:
    return TensorSpec(
        description="This rank's share of the MoE's output.",
        shape=("n_tokens", "model_dim"),
        dtype=DTypeSpec(same_as_tensor="hidden_states"),
        error_bounds=SqnrBound(min_sqnr_db=min_sqnr_db),
    )


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


MINIMAX_M2_FP8_MOE = OpContract(
    contract_id="minimax_m2_fp8_moe@1.0",
    description=(
        f"sglang's MiniMaxM2MoE.forward_normal on one rank with fp8 experts: {_ROUTING}; "
        "each expert down(silu(gate(x)) * up(x)) of its fp8 weights at a scale per "
        "128 x 128 block, each of its two linears taking its input as the checkpoint's FP8 "
        "scheme quantizes it: in bf16, then each 1 x 128 group in fp8 at a scale of its "
        f"amax / 448; {_RANK}."
    ),
    op_type="minimax_m2",
    axes=_axes(
        {
            "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
            "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
            "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
        }
    ),
    attrs=_ATTRS,
    constraints=_CONSTRAINTS,
    inputs={
        **_ROUTING_INPUTS,
        "w1": TensorSpec(
            description="Gate-up projection per expert held: gate rows, then up rows.",
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
            description="Down projection per expert held.",
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
        "expert_offset": _EXPERT_OFFSET,
    },
    # the fp8 activations come to about 36 dB on the bench's inputs
    outputs={"out": _output(30.0)},
    # sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them, the
    # scales as the checkpoint stores them
    layout_variants=_shuffled("w1", "w2"),
    compare=_COMPARE,
)

MINIMAX_M2_MXFP4_MOE = OpContract(
    contract_id="minimax_m2_mxfp4_moe@1.0",
    description=(
        f"sglang's MiniMaxM2MoE.forward_normal on one rank with MXFP4 experts: {_ROUTING}; "
        "each expert down(silu(gate(x)) * up(x)) of its MXFP4 weights (fused_moe's: fp4 "
        f"values at an e8m0 scale per 32); {_RANK}."
    ),
    op_type="minimax_m2",
    axes=_axes(
        {
            "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
            "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
        }
    ),
    attrs=_ATTRS,
    constraints=(
        *_CONSTRAINTS,
        # makes the scales' padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        **_ROUTING_INPUTS,
        **_MXFP4_EXPERTS,
        "expert_offset": _EXPERT_OFFSET,
    },
    # the experts' MXFP4 activations alone come to about 13 dB on the bench's inputs
    outputs={"out": _output(10.0)},
    layout_variants=_MXFP4_LAYOUTS,
    compare=_COMPARE,
)

CONTRACTS = (MINIMAX_M2_FP8_MOE, MINIMAX_M2_MXFP4_MOE)
