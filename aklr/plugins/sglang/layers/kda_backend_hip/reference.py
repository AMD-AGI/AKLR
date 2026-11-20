# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The chain verify from its math, in float32 between its rounding points; independent of the
kernels and of sglang's.

Rounding points: each convolution output, the SiLU of the sum of the four taps' products, to BF16,
which the rings keep as the raw values and keys. The norms, gates, betas and the recurrence, serial
from the slot's state over the chain, run in float32; the outputs and the rings' gates and betas
round to their dtypes in compare."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32
from aklr.plugins.sglang.layers.kda_backend_hip.contract import HEAD_DIM

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kda_chain_verify_ring_reference"]

_NORM_EPS = 1e-6


def _gates(args: Mapping[str, Any], point: Any) -> tuple[Tensor, Tensor]:
    """g (n_tokens, n_heads, HEAD_DIM) and beta (n_tokens, n_heads)."""
    import torch

    dtypes = point.dtypes
    x = to_float32(args["a"], dtypes["a"]) + to_float32(args["dt_bias"], dtypes["dt_bias"])
    scale = torch.exp(to_float32(args["a_log"], dtypes["a_log"]))[:, None]

    def scaled(y: Tensor) -> Tensor:
        # exp(a_log) may pass float32: inf * 0 is 0 here, not nan
        return torch.where(y == 0, 0.0, scale * y)

    if point.attrs["has_lower_bound"]:
        g = args["lower_bound"] * torch.sigmoid(scaled(x))
    else:
        beta, threshold = args["softplus_beta"], args["softplus_threshold"]
        bx = beta * x
        g = -scaled(torch.where(bx <= threshold, torch.log1p(torch.exp(bx)) / beta, x))
    return g, torch.sigmoid(to_float32(args["b"], dtypes["b"]))


def _l2_normalized(x: Tensor) -> Tensor:
    """x * rsqrt(sum(x**2) + eps) over the last dim. Scaled down by a power of two, which rounds as
    unscaled, the squares cannot pass float32 where the norm does not."""
    import torch

    shift = -torch.frexp(x.abs().amax(-1, keepdim=True)).exponent.clamp_min(0)
    x = torch.ldexp(x, shift)
    eps = torch.ldexp(torch.full_like(x[..., :1], _NORM_EPS), 2 * shift)
    return x * (x.square().sum(-1, keepdim=True) + eps).rsqrt()


def kda_chain_verify_ring_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
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
        return to_float32(args[name], dtypes[name])

    # to_float32 may return its input: the mutable tensors are written in copies
    written = {n: read(n).clone() for n in ("conv_window", "ring_v", "ring_k", "ring_g", "ring_beta")}
    x, weight, conv_state, state = (read(n) for n in ("mixed_qkv", "conv_weight", "conv_state", "ssm_state"))
    g, beta = _gates(args, point)
    output = torch.zeros((x.shape[0], heads, HEAD_DIM), device=x.device)
    for r, (slot, line) in enumerate(zip(slots, lines, strict=True)):
        if slot < 0:
            continue
        tokens = slice(r * steps, (r + 1) * steps)
        inputs = torch.cat((conv_state[slot], x[tokens]))  # (steps + 3, conv_dim), oldest first
        written["conv_window"][line] = inputs.unfold(0, 3, 1)[1:].transpose(-1, -2)
        convolved = F.silu((inputs.unfold(0, 4, 1) * weight).sum(-1)).to(torch.bfloat16).float()
        q, k, v = convolved.view(steps, 3, heads, HEAD_DIM).unbind(1)
        written["ring_v"][slot, :, :steps] = v.transpose(0, 1)
        written["ring_k"][slot, :, :steps] = k.transpose(0, 1)
        written["ring_g"][slot, :, :steps] = g[tokens].transpose(0, 1)
        written["ring_beta"][slot, :, :steps] = beta[tokens].T
        q = _l2_normalized(q) * args["scale"]
        k = _l2_normalized(k)
        s = state[slot]  # (heads, value, key)
        for t in range(steps):
            s = s * torch.exp(g[r * steps + t])[:, None, :]
            update = (v[t] - (s * k[t][:, None, :]).sum(-1)) * beta[r * steps + t][:, None]
            s = s + update[..., None] * k[t][:, None, :]
            output[r * steps + t] = (s * q[t][:, None, :]).sum(-1)
    return output, *written.values()
