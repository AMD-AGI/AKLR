# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The chain verify from its math, in float32 between its rounding points; independent of the
kernels and of sglang's.

Rounding points: each convolution output, the SiLU of the sum of the four taps' products, to BF16,
which the rings keep as the raw values and keys. The norms, gates, betas and the recurrence, serial
from the slot's state over the chain, run in float32; the output, the windows and the rings round
to their dtypes."""

import torch
import torch.nn.functional as F

from aklr.core.contract.reference import OpReference

__all__ = ["HEAD_DIM", "KdaChainVerifyRingReference"]

HEAD_DIM = 128  # key and value head width
_NORM_EPS = 1e-6


def _gates(a, b, a_log, dt_bias, lower_bound, softplus_beta, softplus_threshold):
    """g (n_tokens, n_heads, HEAD_DIM) and beta (n_tokens, n_heads)."""
    x = a.float() + dt_bias.float()
    scale = torch.exp(a_log.float())[:, None]

    def scaled(y):
        # exp(a_log) may pass float32: inf * 0 is 0 here, not nan
        return torch.where(y == 0, 0.0, scale * y)

    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(scaled(x))
    else:
        bx = softplus_beta * x
        g = -scaled(torch.where(bx <= softplus_threshold, torch.log1p(torch.exp(bx)) / softplus_beta, x))
    return g, torch.sigmoid(b.float())


def _l2_normalized(x):
    """x * rsqrt(sum(x**2) + eps) over the last dim. Scaled down by a power of two, which rounds as
    unscaled, the squares cannot pass float32 where the norm does not."""
    shift = -torch.frexp(x.abs().amax(-1, keepdim=True)).exponent.clamp_min(0)
    x = torch.ldexp(x, shift)
    eps = torch.ldexp(torch.full_like(x[..., :1], _NORM_EPS), 2 * shift)
    return x * (x.square().sum(-1, keepdim=True) + eps).rsqrt()


class KdaChainVerifyRingReference(OpReference):
    def reference(
        self,
        *,
        mixed_qkv,
        conv_weight,
        conv_state,
        cache_indices,
        conv_window,
        window_indices,
        a,
        b,
        a_log,
        dt_bias,
        ssm_state,
        ring_v,
        ring_k,
        ring_g,
        ring_beta,
        scale,
        lower_bound=None,
        softplus_beta=None,
        softplus_threshold=None,
        **_,
    ):
        """The output, then the convolution windows and the rings as the call leaves them."""
        steps, heads = conv_window.shape[1], b.shape[1]
        slots, lines = cache_indices.tolist(), window_indices.tolist()
        live = [s for s in slots if s >= 0]
        if len(set(live)) != len(live) or len(set(lines)) != len(lines):
            raise ValueError("slots or window lines repeat")

        mutable = {
            "conv_window": conv_window,
            "ring_v": ring_v,
            "ring_k": ring_k,
            "ring_g": ring_g,
            "ring_beta": ring_beta,
        }
        # .float() may return its input: the mutable tensors are written in copies
        written = {n: t.float().clone() for n, t in mutable.items()}
        x, weight, conv_state, state = (
            mixed_qkv.float(),
            conv_weight.float(),
            conv_state.float(),
            ssm_state.float(),
        )
        g, beta = _gates(a, b, a_log, dt_bias, lower_bound, softplus_beta, softplus_threshold)
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
            q = _l2_normalized(q) * scale
            k = _l2_normalized(k)
            s = state[slot]  # (heads, value, key)
            for t in range(steps):
                s = s * torch.exp(g[r * steps + t])[:, None, :]
                update = (v[t] - (s * k[t][:, None, :]).sum(-1)) * beta[r * steps + t][:, None]
                s = s + update[..., None] * k[t][:, None, :]
                output[r * steps + t] = (s * q[t][:, None, :]).sum(-1)
        return {
            "output": output.to(torch.bfloat16),
            **{n: t.to(mutable[n].dtype) for n, t in written.items()},
        }
