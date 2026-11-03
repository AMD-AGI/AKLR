# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fallback checks the MoE routes share; shapes, dtypes and empty calls are the ops' to decline."""

from __future__ import annotations

__all__ = [
    "experts_fall_back",
    "experts_format",
    "runtime_falls_back",
    "topk_falls_back",
]


def runtime_falls_back() -> bool:
    """Whether sglang's graph mode or config keeps MoE calls on sglang."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.runtime_context import get_exec, get_parallel

    exec_, parallel = get_exec(), get_parallel()
    moe = exec_.moe
    return (
        is_in_tc_piecewise_cuda_graph()
        or is_in_breakable_cuda_graph()
        or not get_moe_a2a_backend().is_none()
        or moe.ep_join_mode is not None
        or parallel.dwdp_size > 1
        or exec_.deterministic.enable_deterministic_inference
        or moe.enable_waterfill
        or moe.enable_eplb
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts > 0
        or moe.expert_distribution_recorder_mode is not None
        or exec_.features.enable_return_routed_experts
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def topk_falls_back(topk, *, scoring_func: str, biased: bool) -> bool:
    """Whether ``topk`` routes other than the ops' plain ungrouped top-k of ``scoring_func``
    scores, with a correction bias exactly when ``biased``."""
    config = topk.topk_config
    return (
        config.use_grouped_topk
        or config.custom_routing_function is not None
        or config.scoring_func != scoring_func
        or (config.correction_bias is not None) != biased
        or not config.renormalize
        or config.num_fused_shared_experts != 0
        or config.routed_scaling_factor not in (None, 1.0)
        or config.apply_routed_scaling_factor_on_output
    )


def experts_format(experts) -> str:
    """The class naming ``experts``' format, by sglang's rule: their scheme's, else their quant
    method's."""
    return type(getattr(experts, "scheme", None) or experts.quant_method).__name__


def experts_fall_back(experts, formats) -> bool:
    """Whether ``experts`` run other than as the ops take them: in none of ``formats``, or not as
    gated silu on aiter's fused_moe at sglang's standard dispatch."""
    if (
        experts_format(experts) not in formats
        or type(experts).__name__ != "FusedMoE"
        or type(experts.dispatcher).__name__ != "StandardDispatcher"
        or experts.reduce_results
    ):
        return True
    runner = _supported_runner(experts)
    if runner is None or not runner.runner_backend.is_aiter():
        return True
    config = runner.config
    return (
        config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or config.gemm1_alpha is not None
        or config.gemm1_clamp_limit is not None
        # aiter clamps only at a limit other than 0
        or config.swiglu_limit not in (None, 0.0)
        or getattr(experts, "w13_weight_bias", None) is not None
        or getattr(experts, "w2_weight_bias", None) is not None
    )


def _supported_runner(experts):
    """The runner of ``experts`` held as the ops take them, else None: fp8 at a scale per
    128 x 128 block, or Quark's MXFP4 at dynamic input scales."""
    format_ = experts_format(experts)
    if format_ == "Fp8MoEMethod":
        method = experts.quant_method
        if (
            not method.block_quant
            or method.use_mxfp8
            or method.is_fp4_expert
            or list(method.weight_block_size) != [128, 128]
        ):
            return None
        return getattr(method, "runner", None)
    if format_ == "QuarkW4A4MXFp4MoE":
        scheme = experts.scheme
        if (
            # KTransformers' offload wraps Quark's method and holds only the GPU experts' weights
            type(experts.quant_method).__name__ != "QuarkFusedMoEMethod"
            or scheme.static_input_scales
            # sglang's loader zeroes a padded intermediate, which leaves the ops' sum sglang's; it
            # skips that for presharded weights
            or (getattr(experts, "intermediate_pad", 0) != 0 and experts.use_presharded_weights)
        ):
            return None
        return getattr(scheme, "runner", None)
    return None
