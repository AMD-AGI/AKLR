# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["MhcFusedPostPreReference", "MhcPostReference", "MhcPreReference"]


def _pre(
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_iters,
    norm_weight=None,
    norm_eps=None,
    **_,
):
    n_tokens, streams, _ = residual.shape
    flat = residual.flatten(1)
    rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = (flat @ proj_weight.float().mT) * rms
    mix_scale, mix_bias = mix_scale.float(), mix_bias.float()
    sizes = (streams, streams, streams * streams)
    pre_logits, post_logits, comb_logits = mixes.split(sizes, dim=-1)
    pre_bias, post_bias, comb_bias = mix_bias.split(sizes)
    pre_mix = torch.sigmoid(pre_logits * mix_scale[0] + pre_bias)
    pre_mix = pre_mix + pre_eps
    post_mix = torch.sigmoid(post_logits * mix_scale[1] + post_bias)
    post_mix = post_mix * post_multiplier
    comb_mix = (comb_logits * mix_scale[2] + comb_bias).reshape(n_tokens, streams, streams)
    comb_mix = comb_mix.softmax(dim=-1) + sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = (residual * pre_mix.unsqueeze(-1)).sum(dim=1)
    if norm_weight is not None:
        mean_square = layer_input.square().mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_square + norm_eps)
        layer_input = layer_input * rms * norm_weight.float()
    return post_mix.unsqueeze(-1), comb_mix, layer_input


def _post(x, residual, post_mix, comb_mix):
    residual = residual.float()
    post = x.float().unsqueeze(1) * post_mix.float().reshape(*residual.shape[:2], 1)
    return post + comb_mix.float().mT @ residual


class MhcPreReference(OpReference):
    def reference(self, *, residual, **call):
        """Rms-scaled projection of the residual streams, then sigmoid and Sinkhorn mixes."""
        post_mix, comb_mix, layer_input = _pre(residual.float(), **call)
        return {"post_mix": post_mix, "comb_mix": comb_mix, "layer_input": layer_input.to(residual.dtype)}


class MhcPostReference(OpReference):
    def reference(self, *, x, residual, post_mix, comb_mix, **_):
        """Out[t, j] = x[t] * post_mix[t, j] + sum_i comb_mix[t, i, j] * residual[t, i]."""
        return {"out": _post(x, residual, post_mix, comb_mix).to(residual.dtype)}


class MhcFusedPostPreReference(OpReference):
    def reference(self, *, x, residual, post_mix, comb_mix, **call):
        """Post, then the next pre on its result; with norm_weight, an rms norm of layer_input."""
        next_residual = _post(x, residual, post_mix, comb_mix).to(residual.dtype)
        next_post_mix, next_comb_mix, layer_input = _pre(next_residual.float(), **call)
        return {
            "next_post_mix": next_post_mix,
            "next_comb_mix": next_comb_mix,
            "layer_input": layer_input.to(residual.dtype),
            "next_residual": next_residual,
        }
