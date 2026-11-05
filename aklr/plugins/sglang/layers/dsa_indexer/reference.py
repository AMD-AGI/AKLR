# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch. Each step is computed exactly (float64) and rounded once
where sglang rounds it, the roundings every correct kernel makes:

- the projections q_lora wq_b^T, x wk^T and x weights_proj^T, to bf16;
- k_norm's LayerNorm, to bf16;
- RoPE's turned pairs, to bf16;
- the Hadamard transform (times 128^-0.5), to bf16;
- a head weight times n_heads^-0.5, to bf16;
- the fp8 values at their power-of-two scale, which compare rounds.

A kernel that sums a projection in another order still rounds a few values to the next bf16,
and wherever a row's amax then crosses a power of two times 448, its fp8 scale is twice or half
the reference's and its fp8 values half or twice. Each mutable tensor is therefore bounded by
SQNR, not per value."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.sglang.layers.dsa_indexer.contract import (
    FP8_MAX,
    HEAD_DIM,
    PAGE_SIZE,
    ROPE_DIM,
    SCALE_FLOOR,
)

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["dsa_indexer_key_prepare_reference", "dsa_indexer_prepare_reference"]

# a page: PAGE_SIZE keys of HEAD_DIM fp8 values in 16 x 16 tiles, then their fp32 scales
_TILE = 16
_PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)


def _bf16(t: Tensor) -> Tensor:
    """``t`` rounded to bf16, in float64."""
    import torch

    return t.to(torch.bfloat16).double()


def _project(a: Tensor, w: Tensor) -> Tensor:
    """a w^T, the exact sum rounded once to bf16."""
    return _bf16(a @ w.mT)


def _hadamard(v: Tensor) -> Tensor:
    """The Sylvester-ordered Hadamard transform over the last dim (fast_hadamard_transform's
    order), unscaled: butterflies of stride 1, 2, ..., exact in float64 on bf16 values."""
    import torch

    n, lead = v.shape[-1], v.shape[:-1]
    stride = 1
    while stride < n:
        pairs = v.reshape(*lead, n // (2 * stride), 2, stride)
        first, second = pairs[..., 0, :], pairs[..., 1, :]
        v = torch.stack((first + second, first - second), dim=-2).reshape(*lead, n)
        stride *= 2
    return v


def _rotate(v: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """RoPE on the first ROPE_DIM values in interleaved pairs (2i, 2i + 1), each turned value
    rounded to bf16, then the Hadamard transform times HEAD_DIM^-0.5, rounded to bf16. ``v`` is
    [..., HEAD_DIM] bf16 values in float64; cos and sin [..., ROPE_DIM // 2], broadcast."""
    import torch

    pairs = v[..., :ROPE_DIM].reshape(*v.shape[:-1], ROPE_DIM // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    turned = torch.stack((_bf16(even * cos - odd * sin), _bf16(odd * cos + even * sin)), dim=-1)
    v = torch.cat((turned.reshape(*v.shape[:-1], ROPE_DIM), v[..., ROPE_DIM:]), dim=-1)
    return _bf16(_hadamard(v) * HEAD_DIM**-0.5)


def _scales(v: Tensor) -> Tensor:
    """Each row's fp8 scale: 2^ceil(log2(max(amax, SCALE_FLOOR) / FP8_MAX)), exact."""
    import torch

    ratio = v.abs().amax(-1).clamp(min=SCALE_FLOOR) / FP8_MAX
    mantissa, exponent = torch.frexp(ratio)
    # ratio = mantissa * 2^exponent, mantissa in [0.5, 1): a power of two is mantissa 0.5
    return torch.ldexp(torch.ones_like(ratio), exponent - (mantissa == 0.5).to(exponent.dtype))


def _keys(point: Any, args: Mapping[str, Any], decode: Callable) -> tuple[Tensor, Tensor]:
    """(keys [n_tokens, HEAD_DIM], their scales [n_tokens]): the turned, transformed bf16
    keys in float64, and their fp8 scales."""
    import torch

    dtypes = point.dtypes

    def value(name: str) -> Tensor:
        return decode(args[name], dtypes[name]).double()

    k = _project(value("x"), value("wk"))
    mean = k.mean(-1, keepdim=True)
    variance = (k - mean).square().mean(-1, keepdim=True)
    k = _bf16(
        (k - mean) / torch.sqrt(variance + args["k_norm_eps"]) * value("k_norm_weight") + value("k_norm_bias")
    )
    cos, sin = _rotary(args, decode, dtypes)
    k = _rotate(k, cos, sin)
    return k, _scales(k)


def _rotary(args: Mapping[str, Any], decode: Callable, dtypes: Mapping[str, str]) -> tuple[Tensor, Tensor]:
    """Each token's rotary cos and sin rows [n_tokens, ROPE_DIM // 2], in float64."""
    positions = args["positions"].long()
    cos = decode(args["cos_cache"][positions], dtypes["cos_cache"]).double()
    sin = decode(args["sin_cache"][positions], dtypes["sin_cache"]).double()
    return cos, sin


def _stored_cache(
    point: Any,
    args: Mapping[str, Any],
    decode: Callable,
    keys: Tensor,
    scales: Tensor,
) -> Tensor:
    """index_k_cache after the call, its bytes as fp8 values in float64: each token's fp8
    values (keys / scale, unrounded) and the bytes of its fp32 scale at its slot."""
    import torch

    cache = args["index_k_cache"]
    pages = decode(cache, point.dtypes["index_k_cache"]).double().reshape(-1, _PAGE_BYTES).clone()
    slots = args["out_cache_loc"].long()
    page, key = slots // PAGE_SIZE, slots % PAGE_SIZE
    d = torch.arange(HEAD_DIM)
    # aiter's preshuffled page: [key // 16, d // 16, key % 16, d % 16], then the scales
    columns = (key // _TILE * (_TILE * HEAD_DIM) + key % _TILE * _TILE)[:, None] + (
        d // _TILE * (_TILE * _TILE) + d % _TILE
    )[None, :]
    pages[page[:, None], columns] = keys / scales[:, None]
    scale_bytes = scales.to(torch.float32)[:, None].view(torch.uint8).view(torch.float8_e4m3fn).double()
    scale_columns = (PAGE_SIZE * HEAD_DIM + key * 4)[:, None] + torch.arange(4)[None, :]
    pages[page[:, None], scale_columns] = scale_bytes
    return pages.reshape(cache.shape)


def dsa_indexer_prepare_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(q_fp8, head_weights, index_k_cache): each query head bf16(q_lora wq_b^T), turned and
    transformed as the key, over its scale; head_weights bf16(bf16(x weights_proj^T) *
    n_heads^-0.5) * that scale * HEAD_DIM^-0.5; the keys stored at their slots."""
    dtypes = point.dtypes
    args = row_major_args
    n_heads = point.axes["n_heads"]

    def value(name: str) -> Tensor:
        return decode(args[name], dtypes[name]).double()

    keys, key_scales = _keys(point, args, decode)
    q = _project(value("q_lora"), value("wq_b")).reshape(-1, n_heads, HEAD_DIM)
    cos, sin = _rotary(args, decode, dtypes)
    q = _rotate(q, cos[:, None], sin[:, None])
    q_scales = _scales(q)
    gate = _bf16(_project(value("x"), value("weights_proj")) * n_heads**-0.5)
    head_weights = gate * q_scales * HEAD_DIM**-0.5
    cache = _stored_cache(point, args, decode, keys, key_scales)
    return q / q_scales[..., None], head_weights, cache


def dsa_indexer_key_prepare_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(no_values, index_k_cache): nothing, and the keys stored at their slots."""
    import torch

    keys, key_scales = _keys(point, row_major_args, decode)
    cache = _stored_cache(point, row_major_args, decode, keys, key_scales)
    return torch.zeros(0, dtype=torch.float64), cache
