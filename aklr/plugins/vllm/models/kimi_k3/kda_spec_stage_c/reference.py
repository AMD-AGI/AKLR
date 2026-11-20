# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The stage from vLLM's own ops, in their order and in float32, rounded to bf16 where vLLM hands
a bf16 tensor from one kernel to the next: f_b_proj's GEMM (raw_g), ``causal_conv1d_update`` with the accepted counts
and SiLU (its bf16 output), the speculative recurrence of ``fused_recurrent_kda`` into vLLM's bf16 core_attn_out,
then ``FusedRMSNormGated`` with the sigmoid."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.contract import L2_NORM_EPS

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_spec_stage_c_reference", "spec_conv_update"]


def _bf16(x: Tensor) -> Tensor:
    """x rounded to bf16, in x's dtype: a bf16 tensor vLLM writes and its next kernel reads."""
    import torch

    return x.to(torch.bfloat16).to(x.dtype)


def spec_conv_update(
    x: Tensor, weight: Tensor, conv_state: Tensor, slots: list[int], accepted: list[int], live: list[bool]
) -> Tensor:
    """vLLM's ``causal_conv1d_update`` of windows of w = len(x) // len(slots) rows with SiLU, its output rounded to
    bf16: window s reads conv_state[slots[s], a .. a + width - 2] (a = accepted[s] - 1), then its inputs, and leaves
    the history after a, then its inputs, in the slot. ``conv_state`` (slot, conv_len, channel) is written in place
    where the slot is positive and the window ``live``; a window not live gives zeros. x: (rows, channel)."""
    import torch

    width = weight.shape[1]
    window = x.shape[0] // len(slots)
    y = torch.zeros_like(x)
    for s, (slot, n_acc, is_live) in enumerate(zip(slots, accepted, live, strict=True)):
        if not is_live:
            continue
        rows = slice(s * window, (s + 1) * window)
        a = n_acc - 1
        history = conv_state[max(slot, 0), a : a + width - 1]
        taps = torch.cat((history, x[rows]))  # (width - 1 + window, channel)
        z = taps[0:window] * weight[:, 0]
        for j in range(1, width):  # vLLM's order: acc += x_j * w_j, tap by tap
            z = z + taps[j : j + window] * weight[:, j]
        y[rows] = _bf16(z / (1 + torch.exp(-z)))  # vLLM's SiLU
        if slot > 0:
            conv_state[slot] = torch.cat((history[1:], x[rows]))
    return y


def _spec_recurrence(
    qkv: Tensor,
    raw_g: Tensor,
    raw_beta: Tensor,
    a_log: Tensor,
    g_bias: Tensor,
    state_cache: Tensor,
    slots: list[int],
    accepted: list[int],
    scale: float,
    lower_bound: float,
) -> tuple[Tensor, Tensor]:
    """vLLM's ``fused_recurrent_kda`` on windows of w = len(slots) // len(accepted) tokens, in its order: q and k
    normalized (x * rsqrt(sum(x**2) + eps)), then q scaled; token by token the state decayed, the delta, the output.
    Window s starts from slot slots[s * w + accepted[s] - 1] and writes each token's state to the token's slot where
    that is positive; one whose start slot is <= 0 (vLLM's padding) gives zeros and touches no slot. qkv (rows,
    3 * heads, dim), raw_g (rows, heads, dim), raw_beta (rows, heads, 1), state_cache (slot, heads * dim, dim): (out
    (rows, heads, dim), a copy of state_cache as the windows leave it)."""
    import torch

    heads, dim = g_bias.shape
    q, k, v = qkv.unflatten(1, (3, heads)).unbind(1)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    gate = lower_bound * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    decay, beta = gate.exp(), torch.sigmoid(raw_beta[..., 0])
    state_cache = state_cache.clone()
    states = state_cache.view(-1, heads, dim, dim)  # (slot, head, v, k)
    window = len(slots) // len(accepted)
    out = torch.zeros_like(v)
    for s, n_acc in enumerate(accepted):
        first = s * window
        start = slots[first + n_acc - 1]
        if start <= 0:
            continue
        state = states[start].clone()
        for t in range(first, first + window):
            state = state * decay[t][:, None, :]
            delta = (v[t] - (state * k[t][:, None, :]).sum(-1)) * beta[t][:, None]
            state = state + delta[:, :, None] * k[t][:, None, :]
            out[t] = (state * q[t][:, None, :]).sum(-1)
            if slots[t] > 0:
                states[slots[t]] = state
    return out, state_cache


def kimi_k3_kda_spec_stage_c_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """(out, conv_state, state_cache) as the call leaves them."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    heads, dim = point.axes["n_heads"], point.axes["head_dim"]
    n_seqs, n_tokens = point.axes["n_seqs"], point.axes["n_tokens"]
    window = point.axes["conv_len"] - point.axes["width"] + 2
    rows = n_seqs * window
    accepted = args["num_accepted_tokens"].tolist()
    slots = args["state_indices"][:rows].tolist()
    starts = [slots[s * window + n_acc - 1] for s, n_acc in enumerate(accepted)]
    live = [start > 0 for start in starts]

    # (1) f_b_proj: vLLM's GEMM writes bf16
    f_a = to_float32(args["f_a"], dtypes["f_a"])[:rows, 0]
    raw_g = _bf16(f_a @ to_float32(args["f_b_weight"], dtypes["f_b_weight"]).T).view(rows, heads, dim)
    # (2) the convolution of the windows' packed q|k|v columns
    conv_state = to_float32(args["conv_state"], dtypes["conv_state"]).clone()
    qkv = to_float32(args["qkv"], dtypes["qkv"])[:rows].flatten(1)
    conv_weight = to_float32(args["conv_weight"], dtypes["conv_weight"])
    y = spec_conv_update(
        qkv, conv_weight, conv_state, [slots[s * window] for s in range(n_seqs)], accepted, live
    )
    # (3) the speculative recurrence through each window
    o, state_cache = _spec_recurrence(
        y.view(rows, 3 * heads, dim),
        raw_g,
        to_float32(args["raw_beta"], dtypes["raw_beta"])[:rows],
        to_float32(args["a_log"], dtypes["a_log"]),
        to_float32(args["g_bias"], dtypes["g_bias"]),
        to_float32(args["state_cache"], dtypes["state_cache"]),
        slots,
        accepted,
        args["scale"],
        args["lower_bound"],
    )
    # (4) the gated RMSNorm of vLLM's bf16 core_attn_out
    o = _bf16(o)
    gate = to_float32(args["out_gate"], dtypes["out_gate"])[:rows]
    norm_weight = to_float32(args["norm_weight"], dtypes["norm_weight"])
    rstd = 1 / torch.sqrt((o * o).sum(-1, keepdim=True) / dim + args["eps"])  # FLA's layer_norm_gated, RMS
    normed = o * rstd * norm_weight * torch.sigmoid(gate)
    is_live = torch.tensor(live, device=o.device).repeat_interleave(window)[:, None, None]
    out = torch.zeros((n_tokens, heads, dim), dtype=o.dtype, device=o.device)
    out[:rows] = torch.where(is_live, normed, 0.0)
    return out, conv_state, state_cache
