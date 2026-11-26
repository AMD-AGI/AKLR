# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's top-k of DeepSeek-V4's sqrtsoftplus router with aiter's fused MoE on (``fused_topk_bias`` in
``model_executor/layers/fused_moe/router/fused_topk_bias_router.py``): no shared columns, vLLM's padded rows
marked by a mask."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)
from aklr.plugins.vllm.layers.fused_topk_bias.baseline import DeepseekV4TopkBaseline
from aklr.plugins.vllm.layers.fused_topk_bias.reference import DeepseekV4TopkReference

__all__ = ["CONTRACTS", "DEEPSEEK_V4_TOPK"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
# vLLM's IMAGE_SENTINEL_BASE_ID for DeepSeek-V4.1: tokens _SENTINEL .. _SENTINEL + 4 are image sentinels
_SENTINEL = 129264

DEEPSEEK_V4_TOPK = OpContract(
    contract_id="deepseek_v4_topk@1.0",
    description=(
        "vLLM's top-k of a sqrtsoftplus router whose shared experts run apart (DeepSeek-V4 with aiter's fused "
        "MoE): each token's n_slots experts by sqrt(softplus(router_logits)) + correction_bias, weighed with the "
        "unbiased scores, divided by their sum when renormalize, times routed_scaling_factor; a row whose "
        "input_tokens id is one of the five image sentinels from image_sentinel_lo selects with bias_vl instead. "
        "Rows padding_mask marks, when given, take ids -1 and weights 0."
    ),
    op_type="fused_topk_bias",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 65536, is_var=True),
        "n_experts": axis("Routed experts scored per token.", 1, 1024),
        "n_slots": axis("Experts per token.", 1, 64),
    },
    attrs={
        "renormalize": attr(
            "Whether the weights are divided by their sum.", True, False, param_name="renormalize"
        ),
        "has_padding_mask": attr(
            "Whether padding_mask is passed.", False, True, default=False, abbrev="masked"
        ),
        "has_bias_vl": attr(
            "Whether bias_vl is passed (DeepSeek-V4.1's image sentinels).",
            False,
            True,
            default=False,
            abbrev="vl",
        ),
        "has_input_tokens": attr(
            "Whether input_tokens is passed, which bias_vl reads.",
            False,
            True,
            default=False,
            abbrev="tokens",
        ),
    },
    # one to six experts per lane of a 64-lane wave, as aiter's topk_gating holds them
    constraints=(
        "n_slots <= n_experts",
        "has_bias_vl == has_input_tokens",
        "n_experts == 64 or n_experts == 128 or n_experts == 256 or n_experts == 384",
    ),
    inputs={
        "router_logits": TensorSpec(
            description="Router logits per token.",
            shape=("n_tokens", "n_experts"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32, DType.BFLOAT16)),
            distribution=FiniteDistribution(),
        ),
        "correction_bias": TensorSpec(
            description="Per-expert bias added to the scores for selection only.",
            shape=("n_experts",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32, DType.BFLOAT16)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "padding_mask": TensorSpec(
            description="Rows that are CUDA graph or DP padding (vLLM's is_padding), anywhere in the batch.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.BOOL,)),
            present_when="has_padding_mask",
            distribution=IntRangeDistribution(lo=0, hi=1),
        ),
        "input_tokens": TensorSpec(
            description="Each row's token id: an image sentinel selects its experts with bias_vl.",
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            present_when="has_input_tokens",
            # about half of them DeepSeek-V4.1's five sentinels
            distribution=IntRangeDistribution(lo=_SENTINEL - 5, hi=_SENTINEL + 9),
        ),
        "bias_vl": TensorSpec(
            description="Per-expert bias added to the scores of image sentinel rows, for selection only.",
            shape=("n_experts",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            present_when="has_bias_vl",
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "image_sentinel_lo": ScalarSpec(
            description="The first of the five image sentinel token ids (DeepSeek-V4.1's); 0 without bias_vl.",
            dtype=DType.INT64,
            values=(_SENTINEL, 0),
            default=_SENTINEL,
        ),
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies every weight.",
            dtype=DType.FLOAT32,
            interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
            default=1.0,
        ),
    },
    outputs={
        "topk_weights": TensorSpec(
            description="Per token: the experts' unbiased scores, renormalized and scaled; 0 on marked rows.",
            shape=("n_tokens", "n_slots"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-4),
        ),
        "topk_ids": TensorSpec(
            description="Per token: the experts by descending biased score; -1 on marked rows.",
            shape=("n_tokens", "n_slots"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=DeepseekV4TopkReference(),
    baseline=DeepseekV4TopkBaseline(),
)

CONTRACTS = (DEEPSEEK_V4_TOPK,)
