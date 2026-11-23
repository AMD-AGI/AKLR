# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs for the stage in a pure verify step (``kda.py``'s aiter branch): f_b_proj through vLLM's unquantized
GEMM dispatch on the strided f_a columns (on gfx950 a copy of f_a and a GEMM, or a Triton GEMM on the view), then
aiter's fused convolution, recurrence and gated norm (``rocm_aiter_ops.kda_fused_recurrent_spec_decode``), on the
operands as kda.py hands them over: [1, n, heads, dim] views of the in-projection's rows, the convolution state
transposed to (slot, channel, position) and the convolution weight width-major, which vLLM stages when it loads the
weights (``process_weights_after_loading``): here outside the timed calls, see ``_staged_conv_weight``."""

from __future__ import annotations

import struct
import weakref
from types import SimpleNamespace

__all__ = ["kimi_k3_kda_spec_stage_c_baseline"]

# the width-major copy of each convolution weight tensor, by the tensor's id while it lives (_staged_conv_weight)
_STAGED: dict = {}
# each window's offsets (vLLM's spec_query_start_loc) by (device, sequences, window): values of the shape alone
_OFFSETS: dict = {}


def _float32(x: float) -> float:
    """x as the float32 a kernel receives."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _staged_conv_weight(conv_weight, width):
    """vLLM's decode_conv1d_weight of ``conv_weight``: its (3, width, heads * dim) width-major copy, made when the
    weights are loaded and read by every step, its CUDA graphs included. Here the copy belongs to the weight tensor,
    freed with it (an allocation that reuses the address never finds it), and every call outside a graph capture
    stages it again from the weight's values (validation reloads weights in place, address and version kept). A
    capture, which is what the timed calls replay, reads the copy the eager call before it made and records no copy
    of its own; only a capture of a weight never seen eagerly stages it inside the capture."""
    import torch

    key = id(conv_weight)
    staged = _STAGED.get(key)
    capturing = conv_weight.is_cuda and torch.cuda.is_current_stream_capturing()
    if staged is not None and capturing:
        return staged
    if staged is None:
        staged = torch.empty(
            (3, width, conv_weight.shape[0] // 3), dtype=conv_weight.dtype, device=conv_weight.device
        )
        _STAGED[key] = staged
        weakref.finalize(conv_weight, _STAGED.pop, key, None)
    staged.copy_(conv_weight.view(3, -1, width).transpose(1, 2))
    return staged


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
    # aiter's kernel takes q's scale as head_dim ** -0.5 in float32; the workloads record that float32
    if _float32(scale) != _float32(dim**-0.5):
        raise NotImplementedError("vLLM's fused kernel scales q by head_dim ** -0.5")
    n_seqs = num_accepted_tokens.shape[0]
    width = conv_weight.shape[1]
    window = conv_state.shape[1] - width + 2
    rows = n_seqs * window
    device = qkv.device
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
        conv_weight=_staged_conv_weight(conv_weight, width),
        out_gate=out_gate[None, :rows],
        norm_weight=norm_weight,
        norm_eps=eps,
        out=out[None, :rows],
    )
    if n_tokens > rows:
        out[rows:].zero_()
    return out
