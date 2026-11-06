# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

__all__ = [
    "deepseek_v4_hc_boundary_norm_mxfp8_reference",
    "deepseek_v4_hc_boundary_norm_reference",
    "deepseek_v4_hc_pre_norm_reference",
]


if TYPE_CHECKING:
    from torch import Tensor


_STREAMS = 4

_POST_MULTIPLIER = 2.0

_FP8_MAX = 448.0

_GROUP = 32


def _hc_mixes_and_norm(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
    streams: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(norm, pre, post, comb) of a boundary whose new streams are ``streams`` (float64 holding
    bf16 values): the collapse under pre_prev rounded to bf16, the RMSNorm of that row rounded to
    bf16, and the mixes of the streams in float64."""
    import torch

    def arg(name: str) -> Tensor:
        return decode(row_major_args[name], point.dtypes[name]).double()

    pre_prev, hc_fn, hc_scale, hc_base = arg("pre_prev"), arg("hc_fn"), arg("hc_scale"), arg("hc_base")
    hc_eps = row_major_args["hc_eps"]
    y = rounded_to_bfloat16((pre_prev.unsqueeze(-1) * streams).sum(dim=1))
    flat = streams.flatten(1)
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
    rstd = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + row_major_args["norm_eps"])
    norm = rounded_to_bfloat16(y * rstd * arg("norm_weight"))
    return norm, pre, post, comb


def _hc_pending_post(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """residual_out: post_in[k] * x + sum_j comb_in[j, k] * residual[j], rounded once to bf16."""

    def arg(name: str) -> Tensor:
        return decode(row_major_args[name], point.dtypes[name]).double()

    x, residual, post_in, comb_in = arg("x"), arg("residual"), arg("post_in"), arg("comb_in")
    return rounded_to_bfloat16(post_in.unsqueeze(-1) * x.unsqueeze(1) + comb_in.mT @ residual)


def _mxfp8(norm: Tensor) -> tuple[Tensor, Tensor]:
    """E4M3FN codes of each group of 32 at the smallest power of two >= amax / 448 (amax floored
    at 1e-10, the product rounded to float32, the exponent byte clamped to [1, 254]), and that
    scale, both as float64."""
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
    """(residual_out, norm, pre, post, comb): the deferred hc_post, the collapse under pre_prev,
    this boundary's mixes and the RMSNorm of the collapsed row.

    Rounding points of the op, as sglang's boundary and norm launches store them: residual_out to
    bf16 (the collapse and the mixes read those bf16 streams), the collapse y to bf16 (the norm
    reads that bf16 row) and norm to bf16. Everything between is float64 here; the mixes and the
    norm's sum of squares are error left to compare.
    """
    streams = _hc_pending_post(point, row_major_args, decode)
    norm, pre, post, comb = _hc_mixes_and_norm(point, row_major_args, decode, streams)
    return streams, norm, pre, post, comb


def deepseek_v4_hc_boundary_norm_mxfp8_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(residual_out, norm, quantized, scales, pre, post, comb): deepseek_v4_hc_boundary_norm's
    rounding points, then the MXFP8 operand of the bf16 norm (sglang's fp8_grid_quant): the scale
    amax * float32(1 / 448) rounded to float32, its exponent rounded up, the division by that power
    of two exact and the E4M3FN rounding the quantization itself."""
    streams = _hc_pending_post(point, row_major_args, decode)
    norm, pre, post, comb = _hc_mixes_and_norm(point, row_major_args, decode, streams)
    quantized, scales = _mxfp8(norm)
    return streams, norm, quantized, scales, pre, post, comb


def deepseek_v4_hc_pre_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(norm, pre, post, comb): the collapse of residual under pre_prev, the mixes of residual and
    the RMSNorm of the collapsed row.

    Rounding points of the op: the collapse y to bf16 (sglang's mhc_combine stores it) and norm
    to bf16; the mixes read residual as given.
    """
    streams = decode(row_major_args["residual"], point.dtypes["residual"]).double()
    return _hc_mixes_and_norm(point, row_major_args, decode, streams)
