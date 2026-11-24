# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The adaptations of main's ``aklr/ops`` contracts to the ops aiter routes: ``python -m codegen.aiter
[REV]`` generates the aiter families from main at ``REV`` (``origin/main`` when not given)."""

from __future__ import annotations

import sys

from codegen.adaptation import Adaptation
from codegen.generation import Generation

__all__ = ["ADAPTATIONS", "main"]

# aiter's weight is (n, k), with no bias beside it, its scales K-major
_FP8_GEMM = {"trans_b": True, "has_bias": False}
_BLOCKSCALE = {**_FP8_GEMM, "scale_major_mode": "K"}
_XQ_WQ = {"a": "XQ", "b": "WQ", "a_scale": "x_scale", "b_scale": "w_scale", "out_dtype": "dtype"}
_X_W = {"a": "x", "b": "w", "a_scale": "x_scale", "b_scale": "w_scale", "out_dtype": "dtype", "out": "y"}
_E4M3 = {"dtype": {"any_of": ["float8_e4m3fn"]}}

ADAPTATIONS = (
    Adaptation(
        source="rms_norm@1.0",
        family="rms_norm",
        target="rmsnorm",
        versions=(2,),
        rename={"x": "input"},
        # aiter's op has the param without a default; main's kernels are the plain RMSNorm
        override={
            "attrs": {
                "use_model_sensitive_rmsnorm": {
                    "description": "aiter's model-sensitive variant; 0 is the plain RMSNorm.",
                    "values": [0],
                    "default": 0,
                    "param_name": "use_model_sensitive_rmsnorm",
                    "optional": True,
                }
            }
        },
    ),
    Adaptation(
        source="add_rms_norm@1.0",
        family="rms_norm",
        target="rmsnorm_with_add",
        versions=(1,),
        rename={"x": "input", "residual": "residual_in", "normalized": "out"},
    ),
    Adaptation(
        source="gemm@1.0",
        family="gemm",
        target="gemm_a16w16",
        versions=(1,),
        rename={"a": "A", "b": "B", "out_dtype": "otype"},
        pin={"trans_b": True, "layout_variant": "row_major"},
        facts=("b.is_shuffled",),
    ),
    # aiter's Triton gemm: the same maths under other names, its output bf16 unless dtype says otherwise
    Adaptation(
        source="gemm@1.0",
        family="gemm",
        target="gemm_a16w16_activation",
        versions=(1,),
        rename={"a": "x", "b": "w", "out_dtype": "dtype", "out": "y"},
        pin={"trans_b": True, "layout_variant": "row_major"},
        facts=("b.is_shuffled",),
        override={"outputs": {"y": {"dtype": {"same_as_tensor": None, "default": "bfloat16"}}}},
    ),
    # aiter reads the weight as shuffled whatever it carries, fp8 e4m3 only
    Adaptation(
        source="gemm_a8w8@1.0",
        family="gemm",
        target="gemm_a8w8_bpreshuffle",
        versions=(1,),
        rename=_XQ_WQ,
        pin={**_FP8_GEMM, "layout_variant": "b_shuffled"},
        override={"inputs": {"XQ": _E4M3, "WQ": _E4M3}},
    ),
    Adaptation(
        source="gemm_a8w8_blockwise_scaled@1.0",
        family="gemm",
        target="gemm_a8w8_blockscale",
        versions=(1,),
        rename=_XQ_WQ,
        pin={**_BLOCKSCALE, "layout_variant": "row_major"},
        facts=("b.is_shuffled",),
    ),
    Adaptation(
        source="gemm_a8w8_blockwise_scaled@1.0",
        family="gemm",
        target="gemm_a8w8_blockscale_bpreshuffle",
        versions=(1,),
        rename=_XQ_WQ,
        pin={**_BLOCKSCALE, "layout_variant": "b_shuffled"},
    ),
    # no aiter op takes these: they move as they are, for validation only
    Adaptation(source="bmm@1.0", family="bmm"),
    Adaptation(source="topk_transform_paged@1.0", family="topk_transform_paged"),
    # aiter's Triton blockscale gemm
    Adaptation(
        source="gemm_a8w8_blockwise_scaled@1.0",
        family="gemm",
        target="gemm_a8w8_blockscale_out",
        versions=(1,),
        rename=_X_W,
        pin={**_BLOCKSCALE, "layout_variant": "row_major"},
        facts=("b.is_shuffled",),
    ),
    Adaptation(
        source="biased_grouped_topk@1.0",
        family="topk",
        target="biased_grouped_topk",
        versions=(1,),
        rename={"router_logits": "gating_output", "renormalize": "need_renorm"},
    ),
    # one group, by sigmoid scores: aiter's default score is another, so the contract names its own
    Adaptation(
        source="biased_grouped_topk@1.0",
        family="topk",
        target="topk_gating",
        versions=(1,),
        rename={"router_logits": "gating_output", "topk_ids": "topk_indices", "renormalize": "need_renorm"},
        pin={"num_expert_group": 1, "topk_group": 1},
        override={
            "attrs": {
                "score_func": {
                    "description": "The score of an expert: the sigmoid of its logit.",
                    "values": ["sigmoid"],
                    "default": "sigmoid",
                    "param_name": "score_func",
                    "optional": True,
                }
            }
        },
    ),
    Adaptation(
        source="mhc_pre@1.0",
        family="mhc",
        target="mhc_pre",
        versions=(1,),
        rename={
            "proj_weight": "fn",
            "mix_scale": "hc_scale",
            "mix_bias": "hc_base",
            "pre_eps": "hc_pre_eps",
            "sinkhorn_eps": "hc_sinkhorn_eps",
            "post_multiplier": "hc_post_mult_value",
            "sinkhorn_iters": "sinkhorn_repeat",
        },
    ),
)


def main(argv: list[str]) -> int:
    """Generate the aiter families from main at ``argv[0]``; print main's schema files left behind."""
    left = Generation("aiter", "aklr/ops", "tests/ops/cases.py", ADAPTATIONS).write(
        argv[0] if argv else "origin/main"
    )
    for path in left:
        print(f"not adapted: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
