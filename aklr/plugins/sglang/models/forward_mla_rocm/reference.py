# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in the dtype ``decode`` returns: the inputs decoded
exactly, nothing rounded in between but what sglang's fp8 absorb quantizes, the activation
it multiplies by the fp8 weights (_group_fp8). RoPE, the absorbed products and the qkv
latent stay unrounded (the kernels' bf16 steps are theirs, left to the bounds)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.sglang.models.forward_mla_rocm.contract import FP8_GROUP

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mla_kc_rope_cache_reference", "mla_qkv_a_norm_reference", "mla_vc_reference"]

# fp8 e4m3fn's largest finite value
_FP8_MAX = 448.0


def _absorbed(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    """x [tokens, heads, k] times weight [heads, n, k] and the per-tensor scale, x taken as
    sglang's fp8 absorb quantizes it (aiter's a8w8 batched GEMM, group_size=128): each
    128-wide group of k (the last one maybe narrower) in fp8 e4m3fn at a scale of
    max(amax, 1e-10) / 448, its codes clamped to +-448. The scales and codes are float32's,
    as the kernels compute them: 448 x / amax of bf16 values often falls on an fp8 tie,
    which float32's x * (1 / scale) breaks one way and float64's another."""
    import torch

    out = None
    for start in range(0, x.shape[-1], FP8_GROUP):
        group = x[..., start : start + FP8_GROUP].float()
        group_scale = group.abs().amax(-1, keepdim=True).clamp(min=1e-10) * (1.0 / _FP8_MAX)
        codes = (group * (1.0 / group_scale)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        part = torch.einsum("thk,hnk->thn", codes.to(x.dtype), weight[..., start : start + FP8_GROUP])
        part = part * group_scale.to(x.dtype)
        out = part if out is None else out + part
    return out * scale


def _rms_normalized(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x times the reciprocal rms of its row, plus eps under the root, times weight; the mean
    square in float64, where a float32 square of large finite inputs overflows."""
    mean_square = x.double().square().mean(dim=-1, keepdim=True)
    return x * (mean_square + eps).rsqrt().to(x.dtype) * weight


def mla_qkv_a_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(q_latent, kv_latent, k_rope): each token's hidden state times w_qkv_a, its q and KV
    parts RMS-normalized and scaled by their weights, its RoPE part as projected.
    Rounding points: none (sglang's bf16 qkv latent, before the norms, is left to the bounds)."""
    args = row_major_args

    def get(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name])

    q_rank, kv_rank = point.axes["q_lora_rank"], point.axes["kv_lora_rank"]
    latent = get("hidden_states") @ get("w_qkv_a").T
    eps = args["eps"]
    return (
        _rms_normalized(latent[:, :q_rank], get("q_norm_weight"), eps),
        _rms_normalized(latent[:, q_rank : q_rank + kv_rank], get("kv_norm_weight"), eps),
        latent[:, q_rank + kv_rank :],
    )


def _rotated(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x's GPT-J pairs (2i, 2i + 1) rotated by the angles whose cos and sin are given, one
    per pair."""
    import torch

    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


def mla_kc_rope_cache_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(q_cat, kv_cache): each head's position-free query part absorbed through w_kc, then
    its RoPE part rotated by the token's position; each token's KV latent and rotated RoPE
    key in its KV pool row. A row whose slot is 0 is padding: q_cat 0, no row written.
    Rounding points: the position-free query part's fp8 codes (_absorbed)."""
    import torch

    args = row_major_args

    def get(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name])

    nope, rope = point.axes["qk_nope_dim"], point.axes["kv_dim"] - point.axes["kv_lora_rank"]
    q = get("q")
    positions = args["positions"].long()
    cos, sin = get("cos_cache")[positions], get("sin_cache")[positions]
    q_cat = torch.cat(
        (
            _absorbed(q[..., :nope], get("w_kc"), get("w_scale")),
            _rotated(q[..., nope:], cos[:, None], sin[:, None]),
        ),
        -1,
    )
    slots = args["out_cache_loc"].long()
    written = slots != 0
    q_cat = torch.where(written[:, None, None], q_cat, 0.0)
    rows = torch.cat((get("kv_latent"), _rotated(get("qkv_latent")[:, -rope:], cos, sin)), -1)
    # a copy: decode may return the very tensor it was given
    kv_cache = get("kv_cache").clone()
    kv_cache[slots[written]] = rows[written]
    return q_cat, kv_cache


def mla_vc_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each head's attention output absorbed through w_vc, the heads side by side.
    Rounding points: the attention output's fp8 codes (_absorbed)."""
    args = row_major_args

    def get(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name])

    return _absorbed(get("attn_output"), get("w_vc"), get("w_scale")).flatten(1)
