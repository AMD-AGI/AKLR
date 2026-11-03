# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "mhc_fused_post_pre_reference",
    "mhc_post_reference",
    "mhc_pre_reference",
]


def _decoded(
    point: Any,
    row_major_args: Mapping[str, Any],
    name: str,
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    return decode(row_major_args[name], point.dtypes[name])


def _pre(
    point: Any,
    row_major_args: Mapping[str, Any],
    residual: Tensor,
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    import torch

    n_tokens, streams, _ = residual.shape
    flat = residual.flatten(1)
    proj_weight = _decoded(point, row_major_args, "proj_weight", decode)
    rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + row_major_args["rms_eps"])
    mixes = (flat @ proj_weight.mT) * rms
    mix_scale = _decoded(point, row_major_args, "mix_scale", decode)
    mix_bias = _decoded(point, row_major_args, "mix_bias", decode)
    sizes = (streams, streams, streams * streams)
    pre_logits, post_logits, comb_logits = mixes.split(sizes, dim=-1)
    pre_bias, post_bias, comb_bias = mix_bias.split(sizes)
    pre_mix = torch.sigmoid(pre_logits * mix_scale[0] + pre_bias)
    pre_mix = pre_mix + row_major_args["pre_eps"]
    post_mix = torch.sigmoid(post_logits * mix_scale[1] + post_bias)
    post_mix = post_mix * row_major_args["post_multiplier"]
    sinkhorn_eps = row_major_args["sinkhorn_eps"]
    comb_mix = (comb_logits * mix_scale[2] + comb_bias).reshape(n_tokens, streams, streams)
    comb_mix = comb_mix.softmax(dim=-1) + sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(row_major_args["sinkhorn_iters"] - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = (residual * pre_mix.unsqueeze(-1)).sum(dim=1)
    if point.attrs["has_norm_weight"]:
        norm_weight = _decoded(point, row_major_args, "norm_weight", decode)
        rms = torch.rsqrt(layer_input.square().mean(dim=-1, keepdim=True) + row_major_args["norm_eps"])
        layer_input = layer_input * rms * norm_weight
    return post_mix.unsqueeze(-1), comb_mix, layer_input


def mhc_pre_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Rms-scaled projection of the residual streams, then sigmoid and Sinkhorn mixes."""
    residual = _decoded(point, row_major_args, "residual", decode)
    return _pre(point, row_major_args, residual, decode)


def mhc_post_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Out[t, j] = x[t] * post_mix[t, j] + sum_i comb_mix[t, i, j] * residual[t, i]."""
    residual = _decoded(point, row_major_args, "residual", decode)
    x = _decoded(point, row_major_args, "x", decode)
    post_mix = _decoded(point, row_major_args, "post_mix", decode)
    comb_mix = _decoded(point, row_major_args, "comb_mix", decode)
    post = x.unsqueeze(1) * post_mix.reshape(*residual.shape[:2], 1)
    return post + comb_mix.mT @ residual


def mhc_fused_post_pre_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Post, then the next pre on its result; with norm_weight, an rms norm of layer_input."""
    next_residual = mhc_post_reference(point, row_major_args, decode)
    stored = row_major_args["residual"].dtype
    rounded = next_residual.to(stored).to(next_residual.dtype)
    next_post_mix, next_comb_mix, layer_input = _pre(point, row_major_args, rounded, decode)
    return next_post_mix, next_comb_mix, layer_input, next_residual
