# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on aiter on gfx950 for Qwen2MoeSparseMoeBlock's
_forward_router_experts with the shared expert fused into the experts: the routing from
its small-M router of 1 to 8 Qwen3.5 tokens (smallm_router_gfx950, which appends the gated
shared expert itself), else the gate as ReplicatedLinear runs it, TopK's fused_topk and
_append_shared_to_topk_output's fused gate; then FusedMoE on aiter's runner: sglang's
gfx950 small-M kernel where it serves, else aiter's fused_moe at per_128x128 (which
quantizes the activations per 1 x 128). sglang's runner also patches aiter's MoE sorting
for decode-sized calls (apply_aiter_small_moe_sort_patch); the baselines leave aiter's
process state alone and time aiter's own sorting."""

from __future__ import annotations

import functools

__all__ = [
    "qwen2_moe_fp8_experts_baseline",
    "qwen2_moe_fp8_router_experts_baseline",
    "qwen2_moe_router_baseline",
]

_N_SHARED = 1  # the shared expert fused into the experts' last slot


def _aiter(fn):
    """aiter's own body, not AKLR's stand-in: the speedup is over aiter."""
    proxy = getattr(fn, "aklr_proxy", None)
    return fn if proxy is None else proxy.backend_fn


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baselines replay: it is chosen at
    import, from SGLANG_USE_AITER=1 on gfx950."""
    from sglang.srt.layers.quantization import unquant

    if not (unquant._use_aiter and unquant._is_gfx95):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")


@functools.cache
def _zero_gate(device, width):
    """A shared expert gate of zeros, for the router alone on sglang's small-M router
    (which gates the shared expert in the same launch)."""
    import torch

    return torch.zeros((1, width), dtype=torch.bfloat16, device=device)


def _smallm_router(hidden_states, router_weight, shared_gate_weight):
    """(weights, ids) of sglang's small-M router (qwen_moe_topk_output), the shared expert
    in their last column; None where it does not serve."""
    from sglang.kernels.ops.moe.smallm_router_gfx950 import (
        smallm_router,
        smallm_router_supported,
    )

    if tuple(router_weight.shape) != (512, 4096) or not smallm_router_supported(hidden_states):
        return None
    return smallm_router(hidden_states, router_weight, shared_gate_weight, 1.0)


def _gate(hidden_states, router_weight):
    """The block's gate: ReplicatedLinear's UnquantizedLinearMethod.apply on aiter, aiter's
    triton gemm_a16w16 for the decode shapes _prefer_triton_a16w16 takes, else tgemm.mm.
    """
    from sglang.srt.layers.quantization import unquant

    if unquant._prefer_triton_a16w16(hidden_states, router_weight):
        return unquant._aiter_triton_a16w16(hidden_states, router_weight, None, dtype=hidden_states.dtype)
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(hidden_states, router_weight, None, otype=hidden_states.dtype)


def _topk(hidden_states, router_weight, topk):
    """(weights, ids): the gate, then TopK's softmax top-k with renormalize (fused_topk)."""
    from sglang.srt.layers.moe.topk import fused_topk

    logits = _gate(hidden_states, router_weight)
    return fused_topk(hidden_states, logits, topk, renormalize=True)


def _with_shared(hidden_states, weights, ids, shared_gate_weight, n_routed):
    """_append_shared_to_topk_output on aiter: the shared expert in a last column, its gate's
    GEMV and sigmoid in the append kernel."""
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        fused_append_shared_experts_with_weights,
    )

    ids, weights = fused_append_shared_experts_with_weights(
        ids,
        weights,
        None,
        _N_SHARED,
        N=n_routed,
        fuse_gate=True,
        hidden_states=hidden_states,
        gate_weight=shared_gate_weight,
        scale=1.0,
    )
    return weights, ids


def _experts(hidden_states, weights, ids, w1, w1_scale, w2, w2_scale):
    """FusedMoE on aiter's runner (AiterRunnerCore.run): sglang's gfx950 small-M kernel where
    it serves, else aiter's fused_moe at per_128x128 with sglang's sorting policy."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner import aiter as runner

    if runner._SMALLM_MOE_ON:
        from sglang.kernels.ops.moe import smallm_moe_gfx950 as smallm

        try:
            if smallm.smallm_moe_supported(
                hidden_states,
                w1,
                w2,
                ids,
                expert_mask=None,
                doweight_stage1=False,
                activation_is_silu=True,
                has_bias=False,
                a1_scale=None,
                w13_scale=w1_scale,
                w2_scale=w2_scale,
            ):
                out = smallm.smallm_moe_fwd(hidden_states, w1, w2, weights, ids, w1_scale, w2_scale)
                if out is not None:
                    return out
        except smallm.SmallMMoeUnavailable:
            pass
    return _aiter(fused_moe)(
        hidden_states,
        w1,
        w2,
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_128x128,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=runner._MOE_SORTING_DISPATCH_POLICY,
    )


def qwen2_moe_router_baseline(hidden_states, router_weight, topk_weights=None, topk_ids=None):
    """The block's routing: sglang's small-M router where it serves (its shared column
    dropped, a zero gate standing in), else the gate and fused_topk."""
    _check_aiter_path()
    topk = topk_ids.shape[-1]
    routed = _smallm_router(
        hidden_states,
        router_weight,
        _zero_gate(hidden_states.device, hidden_states.shape[-1]),
    )
    if routed is None or topk + _N_SHARED != routed[1].shape[-1]:
        return _topk(hidden_states, router_weight, topk)
    weights, ids = routed
    return weights[:, :topk], ids[:, :topk]


def qwen2_moe_fp8_experts_baseline(
    hidden_states,
    topk_weights,
    topk_ids,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    out_dtype=None,
    out=None,
):
    """_append_shared_to_topk_output's fused gate, then FusedMoE (_experts)."""
    _check_aiter_path()
    weights, ids = _with_shared(hidden_states, topk_weights, topk_ids, shared_gate_weight, w1.shape[0] - 1)
    return _experts(hidden_states, weights, ids, w1, w1_scale, w2, w2_scale)


def qwen2_moe_fp8_router_experts_baseline(
    hidden_states,
    router_weight,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=10,
    out_dtype=None,
    out=None,
):
    """_forward_router_experts: sglang's small-M router where it serves, else the gate,
    fused_topk and the shared expert's fused gate; then FusedMoE (_experts)."""
    _check_aiter_path()
    routed = _smallm_router(hidden_states, router_weight, shared_gate_weight)
    if routed is None or topk + _N_SHARED != routed[1].shape[-1]:
        weights, ids = _topk(hidden_states, router_weight, topk)
        routed = _with_shared(hidden_states, weights, ids, shared_gate_weight, router_weight.shape[0])
    weights, ids = routed
    return _experts(hidden_states, weights, ids, w1, w1_scale, w2, w2_scale)
