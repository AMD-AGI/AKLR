# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on gfx950 for Kimi-K3.

kimi_k3_latent_moe: on aiter, KimiK3MoE._forward_fused up to its
all-reduce, on one stream: the merged front GEMM (_k3_bf16_gemm), the shared experts'
SituAndMul and down GEMM into their slice, then TopK (aiter's biased_grouped_topk on the bf16
logits and the bias cast to bf16, as select_experts hands them) and FusedMoE on aiter's runner
(fused_moe at per_1x32, Situv2 with the gate and up rows apart, the activations in MXFP8 under
AITER_SITUV2_A8W4=1, sglang's Kimi-K3 environment).

kimi_k3_mla_vc_output_gate: rocm_absorb_v_bmm's bf16 torch.bmm into a (tokens, heads, dim)
buffer, flattened, then the o_proj wrapper's fused mla_output_gate (x * sigmoid(gate) where it
does not cover the shapes).

kimi_k3_mla_prefill_gate: forward_extend's extend_attention_fwd on a zero prefix, which casts
Q, K and V to FP8 for Kimi-K3's shapes on gfx950 (SGLANG_TRITON_FP8_PREFILL_ATTN, on by
default), then the fused output gate."""

from __future__ import annotations

import functools

__all__ = [
    "kimi_k3_latent_moe_baseline",
    "kimi_k3_mla_prefill_gate_baseline",
    "kimi_k3_mla_vc_output_gate_baseline",
]


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baseline replays: it is chosen at import,
    from SGLANG_USE_AITER=1 on gfx950."""
    from sglang.srt.layers.quantization import unquant

    if not (unquant._use_aiter and unquant._is_gfx95):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")


def _fp4x2(weight):
    """An MXFP4 weight as Mxfp4MoEMethod._apply_aiter hands it to the runner: a
    float4_e2m1fn_x2 view (where torch has the dtype), as shuffled as the weight."""
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        return weight
    view = weight.view(torch.float4_e2m1fn_x2)
    view.is_shuffled = getattr(weight, "is_shuffled", False)
    return view


def _experts(z, weights, ids, w1, w1_scale, w2, w2_scale, out):
    """FusedMoE with Mxfp4MoEMethod on aiter's runner (AiterRunnerCore.run) for SiTU: aiter's
    fused_moe into ``out``, at sglang's sorting dispatch policy."""
    import os

    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode
    from sglang.srt.layers.moe.moe_runner import aiter as runner

    saved = os.environ.get("AITER_SITUV2_A8W4")
    os.environ["AITER_SITUV2_A8W4"] = "1"
    try:
        extra = {"output": out} if runner._aiter_fused_moe_supports_output() else {}
        result = fused_moe(
            z,
            _fp4x2(w1),
            _fp4x2(w2),
            weights,
            ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale.view(torch.uint8),
            w2_scale=w2_scale.view(torch.uint8),
            gate_mode=GateMode.SEPARATED.value,
            beta=4.0,
            linear_beta=25.0,
            moe_sorting_dispatch_policy=runner._MOE_SORTING_DISPATCH_POLICY,
            **extra,
        )
    finally:
        if saved is None:
            os.environ.pop("AITER_SITUV2_A8W4", None)
        else:
            os.environ["AITER_SITUV2_A8W4"] = saved
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)


@functools.lru_cache(maxsize=64)
def _bias_as(bias_ptr, version, bias, dtype):
    """The bias in the logits' dtype, cast once as TopKConfig.correction_bias_for_dtype caches it."""
    return bias.to(dtype)


def kimi_k3_latent_moe_baseline(
    hidden_states,
    front_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_w2,
    topk=16,
    latent=None,
    shared=None,
):
    """_forward_fused's single-collective branch (no alt stream, k3_ar_fusion off) before its
    all-reduce: _k3_bf16_gemm, then _forward_shared, then _forward_routed."""
    _check_aiter_path()
    import torch
    from sglang.srt.layers.activation import SituAndMul
    from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu
    from sglang.srt.models import kimi_k3

    m, latent_dim = hidden_states.shape[0], w2.shape[1]
    shared_inter_dim = shared_w2.shape[1]
    n_routed = w1.shape[0]
    if latent is None:
        latent = hidden_states.new_empty((m, latent_dim))
    if shared is None:
        shared = hidden_states.new_empty((m, hidden_states.shape[1]))
    front = kimi_k3._k3_bf16_gemm(hidden_states, front_weight)
    gate_up, logits, z = torch.split(front, (2 * shared_inter_dim, n_routed, latent_dim), dim=-1)
    if m > 1 and not kimi_k3._aiter_k3_opt:
        logits = logits.contiguous()
    z = z.contiguous()
    kimi_k3._k3_bf16_gemm(SituAndMul(beta=4.0, linear_beta=25.0)(gate_up), shared_w2, out=shared)
    weights, ids = biased_grouped_topk_gpu(
        hidden_states,
        logits,
        _bias_as(correction_bias.data_ptr(), correction_bias._version, correction_bias, logits.dtype),
        topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    _experts(z, weights, ids, w1, w1_scale, w2, w2_scale, latent)
    return latent, shared


def kimi_k3_mla_vc_output_gate_baseline(latent, vc_weight, gate, output=None):
    """rocm_absorb_v_bmm on bf16 weights, then the fused output gate."""
    import torch
    from sglang.kernels.ops.attention import mla_output_gate

    tokens, heads = latent.shape[0], latent.shape[1]
    buffer = latent.new_empty((tokens, heads, vc_weight.shape[1]))
    torch.bmm(latent.transpose(0, 1), vc_weight.transpose(1, 2), out=buffer.transpose(0, 1))
    x = buffer.flatten(1, 2)
    if mla_output_gate.covered(x, gate):
        result = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
    else:
        result = x * torch.sigmoid(gate)
    if output is None:
        return result
    output.copy_(result)
    return output


@functools.cache
def _fp8_pool(device):
    """A one-slot FP8 latent pool: zero-prefix extend reads none of it, its dtype selects the
    FP8 path."""
    import torch

    pool = torch.zeros((1, 1, 576), dtype=torch.float8_e4m3fn, device=device)
    return pool, pool[..., :512]


def kimi_k3_mla_prefill_gate_baseline(
    q, k, v, gate, qo_indptr, sm_scale=192**-0.5, max_seq_len=8192, output=None
):
    """extend_attention_fwd with forward_extend's arguments on a zero prefix, then the gate."""
    import torch
    from sglang.kernels.ops.attention import mla_output_gate
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    tokens, heads = q.shape[0], q.shape[1]
    k_pool, v_pool = _fp8_pool(q.device)
    attention = q.new_empty((tokens, heads, v.shape[-1]))
    no_prefix = torch.zeros_like(qo_indptr)
    extend_attention_fwd(
        q,
        k.contiguous(),
        v.contiguous(),
        attention,
        k_pool,
        v_pool,
        qo_indptr.to(torch.int64),
        no_prefix,
        no_prefix.to(torch.int64),
        None,
        True,
        None,
        max_seq_len,
        1.0,
        1.0,
        sm_scale,
    )
    x = attention.view(tokens, -1)
    if mla_output_gate.covered(x, gate):
        return mla_output_gate.kimi_k3_mla_output_gate(x, gate)
    return x * torch.sigmoid(gate)
