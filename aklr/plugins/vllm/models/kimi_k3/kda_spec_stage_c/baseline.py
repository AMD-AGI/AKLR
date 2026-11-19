# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs for the stage in a pure verify step (``kda.py``'s aiter branch): f_b_proj through vLLM's unquantized
GEMM dispatch on the strided f_a columns (on gfx950 a copy of f_a and a GEMM, or a Triton GEMM on the view), then
aiter's fused convolution, recurrence and gated norm (``rocm_aiter_ops.kda_fused_recurrent_spec_decode``), on the
operands as kda.py hands them over: [1, n, heads, dim] views of the in-projection's rows, the convolution state
transposed to (slot, channel, position) and the convolution weight width-major, as vLLM stages it at load time (here
once per weight, not in a timed call)."""

from __future__ import annotations

from types import SimpleNamespace

__all__ = ["kimi_k3_kda_spec_stage_c_baseline"]

# the width-major convolution weight by (device, address, heads), and each window's offsets (vLLM's
# spec_query_start_loc) by (device, sequences, window): made once, not in a timed call
_CONV_WEIGHTS: dict = {}
_OFFSETS: dict = {}


def kimi_k3_kda_spec_stage_c_baseline(
    qkv,
    out_gate,
    f_a,
    raw_beta,
    f_b_weight,
    conv_weight,
    conv_state,
    a_log,
    g_bias,
    norm_weight,
    state_cache,
    state_indices,
    num_accepted_tokens,
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    eps=1e-5,
    out_dtype=None,
    out=None,
):
    """vLLM's out [n_tokens, n_heads, head_dim]; both caches are written in place, as in vLLM. Padding rows are
    zeroed (vLLM leaves them unwritten)."""
    import torch
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    n_tokens, heads, dim = out_gate.shape
    if abs(scale - dim**-0.5) > 1e-12:
        raise NotImplementedError("vLLM's fused kernel scales q by head_dim ** -0.5")
    n_seqs = num_accepted_tokens.shape[0]
    width = conv_weight.shape[1]
    window = conv_state.shape[1] - width + 2
    rows = n_seqs * window
    device = qkv.device
    key = (device, conv_weight.data_ptr(), heads)
    if key not in _CONV_WEIGHTS:
        _CONV_WEIGHTS[key] = conv_weight.view(3, heads * dim, width).transpose(1, 2).contiguous()
    offsets_key = (device, n_seqs, window)
    if offsets_key not in _OFFSETS:
        _OFFSETS[offsets_key] = torch.arange(n_seqs + 1, dtype=torch.int32, device=device) * window
    if out is None:
        out = torch.empty(out_gate.shape, dtype=qkv.dtype, device=device)
    # f_b_proj (ColumnParallelLinear, unquantized) as its quant method calls it
    raw_g = dispatch_unquantized_gemm()(SimpleNamespace(weight=f_b_weight), f_a[:rows, 0], f_b_weight, None)
    q, k, v = (qkv[None, :rows, i * heads : (i + 1) * heads] for i in range(3))
    rocm_aiter_ops.kda_fused_recurrent_spec_decode(
        q=q,
        k=k,
        v=v,
        g=raw_g.view(1, rows, heads, dim),
        beta=raw_beta[None, :rows, :, 0],
        A_log=a_log,
        dt_bias=g_bias.flatten(),
        lower_bound=lower_bound,
        initial_state=state_cache.unflatten(1, (heads, dim)),
        cu_seqlens=_OFFSETS[offsets_key],
        ssm_state_indices=state_indices[:rows].view(n_seqs, window),
        num_accepted_tokens=num_accepted_tokens,
        conv_state=conv_state.transpose(1, 2),
        conv_weight=_CONV_WEIGHTS[key],
        out_gate=out_gate[None, :rows],
        norm_weight=norm_weight,
        norm_eps=eps,
        out=out[None, :rows],
    )
    if n_tokens > rows:
        out[rows:].zero_()
    return out
