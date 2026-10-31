# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "mhc_fused_post_pre_reference",
    "mhc_post_reference",
    "mhc_pre_reference",
]


def mhc_pre_reference(
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rms-scaled projection of the residual streams, then sigmoid and Sinkhorn mixes."""
    import torch

    residual = residual.double()
    n_tokens, streams, _ = residual.shape
    flat = residual.flatten(1)
    rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = (flat @ proj_weight.double().mT) * rms
    mix_scale, mix_bias = mix_scale.double(), mix_bias.double()
    sizes = (streams, streams, streams * streams)
    pre_logits, post_logits, comb_logits = mixes.split(sizes, dim=-1)
    pre_bias, post_bias, comb_bias = mix_bias.split(sizes)
    pre_mix = torch.sigmoid(pre_logits * mix_scale[0] + pre_bias) + pre_eps
    post_mix = torch.sigmoid(post_logits * mix_scale[1] + post_bias) * post_multiplier
    comb_mix = (comb_logits * mix_scale[2] + comb_bias).reshape(n_tokens, streams, streams)
    comb_mix = comb_mix.softmax(dim=-1) + sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = (residual * pre_mix.unsqueeze(-1)).sum(dim=1)
    if norm_weight is not None:
        rms = torch.rsqrt(layer_input.square().mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = layer_input * rms * norm_weight.double()
    return post_mix.unsqueeze(-1), comb_mix, layer_input


def mhc_post_reference(x, residual, post_mix, comb_mix) -> Tensor:
    """Out[t, j] = x[t] * post_mix[t, j] + sum_i comb_mix[t, i, j] * residual[t, i]."""
    residual = residual.double()
    return x.double().unsqueeze(1) * post_mix.double() + comb_mix.double().mT @ residual


def mhc_fused_post_pre_reference(
    x,
    residual,
    post_mix,
    comb_mix,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Post, then the next pre on its result; with norm_weight, an rms norm of
    layer_input."""
    next_residual = mhc_post_reference(x, residual, post_mix, comb_mix)
    rounded = next_residual.to(residual.dtype)
    next_post_mix, next_comb_mix, layer_input = mhc_pre_reference(
        rounded,
        proj_weight,
        mix_scale,
        mix_bias,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_multiplier=post_multiplier,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return next_post_mix, next_comb_mix, layer_input, next_residual
