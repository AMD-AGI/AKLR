# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang runs on aiter on gfx950 for DeepseekV2MoE.forward_normal (sglang 0.5.22)."""

from __future__ import annotations

__all__ = ["deepseek_v2_fp8_moe_baseline", "deepseek_v2_mxfp4_moe_baseline"]


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baseline replays, chosen at import."""
    from sglang.srt.layers.quantization import unquant

    if not (unquant._use_aiter and unquant._is_gfx95):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")


def _in_dtype(bias, dtype):
    """``bias`` in ``dtype``, cast once per tensor as sglang caches it
    (TopKConfig.correction_bias_for_dtype): sglang's fp32 bias reaches top-k in bf16."""
    if bias.dtype == dtype:
        return bias
    cast = getattr(bias, "_aklr_in_dtype", None)
    if cast is None or cast.dtype != dtype:
        cast = bias._aklr_in_dtype = bias.to(dtype)
    return cast


def _route(
    hidden_states,
    router_weight,
    correction_bias,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    routed_scaling_factor,
):
    """The router's bf16 logits (MoEGate on HIP: tgemm.mm), then aiter's grouped top-k on them
    with the bias in their dtype: (weights, ids)."""
    import torch
    from aiter import biased_grouped_topk
    from aiter.tuned_gemm import tgemm

    logits = tgemm.mm(hidden_states, router_weight, None, otype=hidden_states.dtype)
    shape, device = (hidden_states.shape[0], topk), hidden_states.device
    weights = torch.empty(shape, dtype=torch.float32, device=device)
    ids = torch.empty(shape, dtype=torch.int32, device=device)
    biased_grouped_topk(
        logits,
        _in_dtype(correction_bias, logits.dtype),
        weights,
        ids,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    return weights, ids


def deepseek_v2_mxfp4_moe_baseline(
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
    out_dtype=None,
    out=None,
):
    """forward_normal on one rank: _route's top-k, sglang's shared-slot append and aiter's
    fused_moe on the MXFP4 experts at sglang's sorting dispatch policy."""
    _check_aiter_path()
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_append_shared_experts
    from sglang.srt.layers.moe.moe_runner.aiter import _MOE_SORTING_DISPATCH_POLICY

    weights, ids = _route(
        hidden_states,
        router_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    ids, weights = fused_append_shared_experts(ids, weights, 1, 1.0, router_weight.shape[0])
    return fused_moe(
        hidden_states,
        w1,
        w2,
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=_MOE_SORTING_DISPATCH_POLICY,
    )


def deepseek_v2_fp8_moe_baseline(
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
    out_dtype=None,
    out=None,
):
    """forward_normal on one rank with fp8 experts at a scale per 128 x 128 block: _route's
    grouped top-k, sglang's shared-slot append and aiter's fused_moe at per_128x128 (it
    quantizes the activations per 1 x 128) at sglang's sorting dispatch policy. Not replayed:
    the small-batch sort patch sglang's aiter runner installs into aiter (process-wide, so it
    would change other ops' baselines too)."""
    _check_aiter_path()
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_append_shared_experts
    from sglang.srt.layers.moe.moe_runner.aiter import _MOE_SORTING_DISPATCH_POLICY

    weights, ids = _route(
        hidden_states,
        router_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    ids, weights = fused_append_shared_experts(ids, weights, 1, 1.0, router_weight.shape[0])
    return fused_moe(
        hidden_states,
        w1,
        w2,
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_128x128,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=_MOE_SORTING_DISPATCH_POLICY,
    )
