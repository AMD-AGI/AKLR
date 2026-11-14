# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on aiter on gfx950 for Qwen3MoeSparseMoeBlock.forward_normal with
fp8 experts: the gate as ReplicatedLinear runs it, TopK's softmax top-k with renormalize
(select_experts' fused_topk, which calls aiter's fused_topk), then FusedMoE on aiter's runner:
aiter's fused_moe at per_128x128 (which quantizes the activations per 1 x 128). The runner's
gfx950 small-M kernel takes only top-10 or top-11 routing at hidden size 4096 (Qwen3.5), never
a Qwen3-MoE call. sglang's runner also patches aiter's MoE sorting for decode-sized calls
(apply_aiter_small_moe_sort_patch); the baseline leaves aiter's process state alone and times
aiter's own sorting."""

from __future__ import annotations

__all__ = ["qwen3_moe_fp8_moe_baseline"]


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baseline replays: it is chosen at
    import, from SGLANG_USE_AITER=1 on gfx950."""
    from sglang.srt.layers.quantization import unquant

    if not (unquant._use_aiter and unquant._is_gfx95):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")


def _gate(hidden_states, router_weight):
    """The block's gate: ReplicatedLinear's UnquantizedLinearMethod.apply on aiter, aiter's
    triton gemm_a16w16 for the decode shapes _prefer_triton_a16w16 takes, else tgemm.mm.
    """
    from sglang.srt.layers.quantization import unquant

    if unquant._prefer_triton_a16w16(hidden_states, router_weight):
        return unquant._aiter_triton_a16w16(hidden_states, router_weight, None, dtype=hidden_states.dtype)
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(hidden_states, router_weight, None, otype=hidden_states.dtype)


def qwen3_moe_fp8_moe_baseline(
    hidden_states,
    router_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=8,
    out_dtype=None,
    out=None,
):
    """forward_normal on one rank: the gate's bf16 logits, fused_topk's renormalized softmax
    top-k, and aiter's fused_moe on the experts at sglang's sorting dispatch policy."""
    _check_aiter_path()
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner.aiter import _MOE_SORTING_DISPATCH_POLICY
    from sglang.srt.layers.moe.topk import fused_topk

    logits = _gate(hidden_states, router_weight)
    weights, ids = fused_topk(hidden_states, logits, topk, renormalize=True)
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
