# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The chain verify from its math, in float64 between its rounding points; independent of the
kernels and of sglang's.

Rounding points: each convolution output, the SiLU of the exact sum of the four taps' products,
to BF16, which the rings keep as the raw values and keys. The norms, gates, betas and the
recurrence, serial from the slot's state over the chain, run in float64; the outputs and the rings'
gates and betas round to their dtypes in compare."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.sglang.layers.kda_backend_hip.contract import HEAD_DIM

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kda_chain_verify_ring_reference"]

_NORM_EPS = 1e-6


def _gates(args: Mapping[str, Any], point: Any, decode: Callable) -> tuple[Tensor, Tensor]:
    """g (n_tokens, n_heads, HEAD_DIM) and beta (n_tokens, n_heads)."""
    import torch

    dtypes = point.dtypes
    x = decode(args["a"], dtypes["a"]).double() + decode(args["dt_bias"], dtypes["dt_bias"]).double()
    scale = torch.exp(decode(args["a_log"], dtypes["a_log"]).double())[:, None]
    if point.attrs["has_lower_bound"]:
        g = args["lower_bound"] * torch.sigmoid(scale * x)
    else:
        beta, threshold = args["softplus_beta"], args["softplus_threshold"]
        bx = beta * x
        g = -scale * torch.where(bx <= threshold, torch.log1p(torch.exp(bx)) / beta, x)
    return g, torch.sigmoid(decode(args["b"], dtypes["b"]).double())


def kda_chain_verify_ring_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, ...]:
    """(output, conv_window, ring_v, ring_k, ring_g, ring_beta)."""
    import torch
    import torch.nn.functional as F

    args, dtypes = row_major_args, point.dtypes
    steps, heads = point.axes["n_verify_tokens"], point.axes["n_heads"]
    slots, lines = args["cache_indices"].tolist(), args["window_indices"].tolist()
    if len(set(s for s in slots if s >= 0)) != sum(s >= 0 for s in slots) or len(set(lines)) != len(lines):
        raise ValueError("slots or window lines repeat")

    def read(name: str) -> Tensor:
        return decode(args[name], dtypes[name]).double()

    # decode may return its input: the mutable tensors are written in copies
    written = {n: read(n).clone() for n in ("conv_window", "ring_v", "ring_k", "ring_g", "ring_beta")}
    x, weight, conv_state, state = (read(n) for n in ("mixed_qkv", "conv_weight", "conv_state", "ssm_state"))
    g, beta = _gates(args, point, decode)
    output = torch.zeros((x.shape[0], heads, HEAD_DIM), dtype=torch.float64, device=x.device)
    for r, (slot, line) in enumerate(zip(slots, lines, strict=True)):
        if slot < 0:
            continue
        tokens = slice(r * steps, (r + 1) * steps)
        inputs = torch.cat((conv_state[slot], x[tokens]))  # (steps + 3, conv_dim), oldest first
        written["conv_window"][line] = inputs.unfold(0, 3, 1)[1:].transpose(-1, -2)
        convolved = F.silu((inputs.unfold(0, 4, 1) * weight).sum(-1)).to(torch.bfloat16).double()
        q, k, v = convolved.view(steps, 3, heads, HEAD_DIM).unbind(1)
        written["ring_v"][slot, :, :steps] = v.transpose(0, 1)
        written["ring_k"][slot, :, :steps] = k.transpose(0, 1)
        written["ring_g"][slot, :, :steps] = g[tokens].transpose(0, 1)
        written["ring_beta"][slot, :, :steps] = beta[tokens].T
        q = q * (q.square().sum(-1, keepdim=True) + _NORM_EPS).rsqrt() * args["scale"]
        k = k * (k.square().sum(-1, keepdim=True) + _NORM_EPS).rsqrt()
        s = state[slot]  # (heads, value, key)
        for t in range(steps):
            s = s * torch.exp(g[r * steps + t])[:, None, :]
            update = (v[t] - (s * k[t][:, None, :]).sum(-1)) * beta[r * steps + t][:, None]
            s = s + update[..., None] * k[t][:, None, :]
            output[r * steps + t] = (s * q[t][:, None, :]).sum(-1)
    results = {"output": output, **written}
    return tuple(decode(results[n], dtypes[n]) for n in results)
