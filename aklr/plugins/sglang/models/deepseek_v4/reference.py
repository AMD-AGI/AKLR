# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the deepseek_v4_hc_boundary_norm contracts say, in torch: inputs decoded to float64, and
nothing rounded in between but what the contract rounds (residual_out, y and norm to bf16)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "deepseek_v4_hc_boundary_norm_mxfp8_reference",
    "deepseek_v4_hc_boundary_norm_reference",
]

_STREAMS = 4
_POST_MULTIPLIER = 2.0
_FP8_MAX = 448.0
_GROUP = 32


def _boundary(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    import torch

    def arg(name: str) -> Tensor:
        return decode(row_major_args[name], point.dtypes[name]).double()

    x, residual = arg("x"), arg("residual")
    post_in, comb_in, pre_prev = arg("post_in"), arg("comb_in"), arg("pre_prev")
    hc_fn, hc_scale, hc_base = arg("hc_fn"), arg("hc_scale"), arg("hc_base")
    norm_weight = arg("norm_weight")
    hc_eps = row_major_args["hc_eps"]

    # 1. the deferred hc_post: new stream k = post_in[k] * x + sum_j comb_in[j, k] * residual[j]
    streams = post_in.unsqueeze(-1) * x.unsqueeze(1) + comb_in.mT @ residual
    residual_out = rounded_to_bfloat16(streams)
    # 2. the collapse under the previous boundary's pre mix
    y = rounded_to_bfloat16((pre_prev.unsqueeze(-1) * residual_out).sum(dim=1))
    # 3. this boundary's mixes from the new streams
    flat = residual_out.flatten(1)
    rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + row_major_args["rms_eps"])
    logits = (flat @ hc_fn.mT) * rsqrt
    sizes = (_STREAMS, _STREAMS, _STREAMS * _STREAMS)
    pre_logits, post_logits, comb_logits = logits.split(sizes, dim=-1)
    pre_base, post_base, comb_base = hc_base.split(sizes)
    pre = torch.sigmoid(pre_logits * hc_scale[0] + pre_base) + hc_eps
    post = _POST_MULTIPLIER * torch.sigmoid(post_logits * hc_scale[1] + post_base)
    comb = (comb_logits * hc_scale[2] + comb_base).reshape(-1, _STREAMS, _STREAMS)
    comb = comb.softmax(dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(row_major_args["sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    # 4. the RMSNorm of y
    rstd = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + row_major_args["norm_eps"])
    norm = rounded_to_bfloat16(y * rstd * norm_weight)
    return residual_out, norm, pre, post, comb


def _mxfp8(norm: Tensor) -> tuple[Tensor, Tensor]:
    """E4M3FN codes of each group of 32 at the smallest power of two >= amax / 448 (amax floored
    at 1e-10, the exponent byte clamped to [1, 254]), and that scale, both as float64."""
    import torch

    rows, columns = norm.shape
    groups = norm.reshape(rows, columns // _GROUP, _GROUP)
    floor = float(torch.tensor(1e-10, dtype=torch.float32))
    amax = groups.abs().amax(-1).clamp_min(floor)
    divisor = float(torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32))
    raw = (amax * divisor).to(torch.float32).contiguous().view(torch.int32)
    exponent = ((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(1, 254)
    scale = (exponent << 23).contiguous().view(torch.float32).double()
    values = (groups / scale.unsqueeze(-1)).to(torch.float32).clamp(-_FP8_MAX, _FP8_MAX)
    return values.to(torch.float8_e4m3fn).double().reshape(rows, columns), scale


def deepseek_v4_hc_boundary_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The boundary, then the RMSNorm of the collapsed row."""
    return _boundary(point, row_major_args, decode)


def deepseek_v4_hc_boundary_norm_mxfp8_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The boundary, the RMSNorm, and the norm's MXFP8 codes and scales."""
    residual_out, norm, pre, post, comb = _boundary(point, row_major_args, decode)
    quantized, scales = _mxfp8(norm)
    return residual_out, norm, quantized, scales, pre, post, comb
