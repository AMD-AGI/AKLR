# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on aiter on gfx950 for MiniMaxM2MoE.forward_normal: the fp32 gate
through UnquantizedLinearMethod (aiter's tgemm.mm), TopK's sigmoid routing with the
correction bias (select_experts' moe_fused_gate) and FusedMoE on aiter's fused_moe runner,
with the expert mask sglang's dispatcher builds under expert parallelism."""

from __future__ import annotations

import functools

__all__ = ["minimax_m2_fp8_moe_baseline", "minimax_m2_mxfp4_moe_baseline"]


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter MoE path the baselines replay: it is chosen at
    import, from SGLANG_USE_AITER=1 on HIP."""
    from sglang.srt.layers.quantization import unquant

    if not unquant._use_aiter:
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on HIP")


@functools.cache
def _expert_mask(device, n_routed: int, first: int, n_held: int):
    """sglang's StandardDispatcher.expert_mask_gpu for an expert-parallel rank: 1 for the
    experts it holds, built once per rank as sglang builds it."""
    import torch

    mask = torch.zeros(n_routed, dtype=torch.int32, device=device)
    mask[first : first + n_held] = 1
    return mask


def _route(hidden_states, router_weight, correction_bias, topk):
    """MiniMaxM2MoE's gate on fp32 activations (aiter's tgemm.mm) and TopK's routing on
    HIP (select_experts -> biased_topk_jit_kernel_impl -> moe_fused_gate): the routing
    weights, then the expert ids."""
    import torch
    from aiter.tuned_gemm import tgemm
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    logits = tgemm.mm(hidden_states.to(torch.float32), router_weight, None, otype=torch.float32)
    weights, ids = moe_fused_gate(
        logits,
        correction_bias,
        topk=topk,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    return weights.to(torch.float32), ids.to(torch.int32)


def _experts(quant_type, hidden_states, router_weight, expert_offset, w1, **kwargs):
    """FusedMoE on aiter's runner (AiterRunnerCore.run): fused_moe with sglang's
    activation and sorting dispatch policy, its expert mask under expert parallelism, no
    padding. Not replayed: the small-batch sort patch sglang's runner installs into aiter
    (process-wide, so it would change other ops' baselines too)."""
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner.aiter import _MOE_SORTING_DISPATCH_POLICY

    n_routed, n_held = router_weight.shape[0], w1.shape[0]
    mask = None if n_held == n_routed else _expert_mask(hidden_states.device, n_routed, expert_offset, n_held)
    return fused_moe(
        hidden_states,
        w1,
        activation=ActivationType.Silu,
        quant_type=quant_type,
        expert_mask=mask,
        moe_sorting_dispatch_policy=_MOE_SORTING_DISPATCH_POLICY,
        **kwargs,
    )


def minimax_m2_fp8_moe_baseline(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    expert_offset=0,
    out_dtype=None,
    out=None,
):
    """forward_normal with fp8 experts: fused_moe at per_128x128 (it quantizes the
    activations per 1 x 128)."""
    _check_aiter_path()
    from aiter import QuantType

    weights, ids = _route(hidden_states, router_weight, correction_bias, topk)
    return _experts(
        QuantType.per_128x128,
        hidden_states,
        router_weight,
        expert_offset,
        w1,
        w2=w2,
        topk_weight=weights,
        topk_ids=ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def minimax_m2_mxfp4_moe_baseline(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    expert_offset=0,
    out_dtype=None,
    out=None,
):
    """forward_normal with MXFP4 experts (QuarkW4A4MXFp4MoE on gfx950): fused_moe at
    per_1x32 with the gate and up rows apart. sglang also passes its intermediate_pad (128
    at TP4 EP1, where the stored width 512 holds 384 columns of the checkpoint); the
    contract computes the stored width, so does this baseline."""
    _check_aiter_path()
    from aiter import QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    weights, ids = _route(hidden_states, router_weight, correction_bias, topk)
    return _experts(
        QuantType.per_1x32,
        hidden_states,
        router_weight,
        expert_offset,
        w1,
        w2=w2,
        topk_weight=weights,
        topk_ids=ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gate_mode=GateMode.SEPARATED.value,
    )
