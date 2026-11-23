# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch. Each step is computed in float32 and rounded where sglang
rounds it, the roundings every correct kernel makes:

- the projections q_lora wq_b^T, x wk^T and x weights_proj^T, to bf16;
- k_norm's LayerNorm, to bf16;
- RoPE's turned pairs, to bf16;
- the Hadamard transform (times 128^-0.5), to bf16;
- a head weight times n_heads^-0.5, to bf16;
- the fp8 values at their power-of-two scale, to fp8.

A kernel that sums a projection in another order still rounds a few values to the next bf16,
and wherever a row's amax then crosses a power of two times 448, its fp8 scale is twice or half
the reference's and its fp8 values half or twice. The outputs are therefore bounded by SQNR,
not per value."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = [
    "FP8_MAX",
    "HEAD_DIM",
    "PAGE_SIZE",
    "ROPE_DIM",
    "SCALE_FLOOR",
    "DsaIndexerKeyPrepareReference",
    "DsaIndexerPrepareReference",
]

# An indexer head and key: the Hadamard transform's length and the cache's 128 bytes a key
HEAD_DIM = 128
# The RoPE part: a head's and a key's first 64 values, in interleaved pairs (2i, 2i + 1)
ROPE_DIM = 64
# Keys per index-K page: aiter's preshuffled layout of 16 x 16 tiles
PAGE_SIZE = 64
# A key's or head's fp8 scale: 2 ** ceil(log2(max(amax, SCALE_FLOOR) / FP8_MAX))
SCALE_FLOOR = 1e-4
FP8_MAX = 448.0

# a page: PAGE_SIZE keys of HEAD_DIM fp8 values in 16 x 16 tiles, then their fp32 scales
_TILE = 16
_PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)


def _bf16(t):
    """``t`` rounded to bf16, in float32."""
    return t.to(torch.bfloat16).float()


def _project(a, w):
    """a w^T, rounded to bf16."""
    return _bf16(a @ w.mT)


def _hadamard(v):
    """The Sylvester-ordered Hadamard transform over the last dim (fast_hadamard_transform's
    order), unscaled: butterflies of stride 1, 2, ...."""
    n, lead = v.shape[-1], v.shape[:-1]
    stride = 1
    while stride < n:
        pairs = v.reshape(*lead, n // (2 * stride), 2, stride)
        first, second = pairs[..., 0, :], pairs[..., 1, :]
        v = torch.stack((first + second, first - second), dim=-2).reshape(*lead, n)
        stride *= 2
    return v


def _rotate(v, cos, sin):
    """RoPE on the first ROPE_DIM values in interleaved pairs (2i, 2i + 1), each turned value
    rounded to bf16, then the Hadamard transform times HEAD_DIM^-0.5, rounded to bf16. ``v`` is
    [..., HEAD_DIM] bf16 values in float32; cos and sin [..., ROPE_DIM // 2], broadcast."""
    pairs = v[..., :ROPE_DIM].reshape(*v.shape[:-1], ROPE_DIM // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    turned = torch.stack((_bf16(even * cos - odd * sin), _bf16(odd * cos + even * sin)), dim=-1)
    v = torch.cat((turned.reshape(*v.shape[:-1], ROPE_DIM), v[..., ROPE_DIM:]), dim=-1)
    return _bf16(_hadamard(v) * HEAD_DIM**-0.5)


def _scales(v):
    """Each row's fp8 scale: 2^ceil(log2(max(amax, SCALE_FLOOR) / FP8_MAX)), exact."""
    ratio = v.abs().amax(-1).clamp(min=SCALE_FLOOR) / FP8_MAX
    mantissa, exponent = torch.frexp(ratio)
    # ratio = mantissa * 2^exponent, mantissa in [0.5, 1): a power of two is mantissa 0.5
    return torch.ldexp(torch.ones_like(ratio), exponent - (mantissa == 0.5).to(exponent.dtype))


def _rotary(positions, cos_cache, sin_cache):
    """Each token's rotary cos and sin rows [n_tokens, ROPE_DIM // 2], in float32."""
    positions = positions.long()
    return cos_cache[positions].float(), sin_cache[positions].float()


def _keys(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, cos, sin):
    """(keys [n_tokens, HEAD_DIM], their scales [n_tokens]): the turned, transformed bf16
    keys in float32, and their fp8 scales."""
    k = _project(x.float(), wk.float())
    mean = k.mean(-1, keepdim=True)
    variance = (k - mean).square().mean(-1, keepdim=True)
    k = _bf16((k - mean) / torch.sqrt(variance + k_norm_eps) * k_norm_weight.float() + k_norm_bias.float())
    k = _rotate(k, cos, sin)
    return k, _scales(k)


def _key_columns(slots):
    """(page [n, 1], value columns [n, HEAD_DIM], scale columns [n, 4]) of each slot's key in a
    page's bytes."""
    slots = slots.long()
    page, key = slots // PAGE_SIZE, slots % PAGE_SIZE
    d = torch.arange(HEAD_DIM, device=slots.device)
    # aiter's preshuffled page: [key // 16, d // 16, key % 16, d % 16], then the scales
    values = (key // _TILE * (_TILE * HEAD_DIM) + key % _TILE * _TILE)[:, None] + (
        d // _TILE * (_TILE * _TILE) + d % _TILE
    )[None, :]
    scales = (PAGE_SIZE * HEAD_DIM + key * 4)[:, None] + torch.arange(4, device=slots.device)[None, :]
    return page[:, None], values, scales


def _stored_cache(index_k_cache, out_cache_loc, keys, scales):
    """index_k_cache after the call: each token's fp8 values (keys / scale) and the bytes of its
    fp32 scale at its slot."""
    pages = index_k_cache.float().reshape(-1, _PAGE_BYTES).clone()
    page, value_columns, scale_columns = _key_columns(out_cache_loc)
    pages[page, value_columns] = keys / scales[:, None]
    pages[page, scale_columns] = scales[:, None].view(torch.uint8).view(torch.float8_e4m3fn).float()
    return pages.reshape(index_k_cache.shape).to(torch.float8_e4m3fn)


def _restated_scale_steps(cache, expected, slots):
    """``cache`` with each key at ``slots`` whose fp32 scale is twice or half ``expected``'s
    restated at ``expected``'s scale: its fp8 values times that ratio (exact but in fp8's
    subnormals, a power of two moving only the exponent) and its scale bytes ``expected``'s."""
    pages = cache.view(torch.uint8).reshape(-1, _PAGE_BYTES)
    expected_pages = expected.view(torch.uint8).reshape(-1, _PAGE_BYTES)
    page, value_columns, scale_columns = _key_columns(slots)
    ratio = (
        pages[page, scale_columns].contiguous().view(torch.float32)
        / expected_pages[page, scale_columns].contiguous().view(torch.float32)
    )[:, 0]
    stepped = ((ratio == 2) | (ratio == 0.5)).nonzero(as_tuple=True)[0]
    if not stepped.numel():
        return cache
    pages = pages.clone()
    page, value_columns, scale_columns = page[stepped], value_columns[stepped], scale_columns[stepped]
    values = pages[page, value_columns].view(torch.float8_e4m3fn).float() * ratio[stepped, None]
    pages[page, value_columns] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    pages[page, scale_columns] = expected_pages[page, scale_columns]
    return pages.view(cache.dtype).reshape(cache.shape)


class _IndexKCacheReference(OpReference):
    def check(self, outputs, expected, schema):
        """``OpReference.check``, but a key stored at twice or half the reference's scale, its fp8
        values half or twice, stands for the reference's key: a kernel that sums a projection in
        another order rounds a few values to the next bf16, and a key's amax can then cross a power
        of two times 448. Any other scale or value still differs."""
        cache = _restated_scale_steps(
            outputs["index_k_cache"], expected["index_k_cache"], expected["written_slots"]
        )
        return super().check({**outputs, "index_k_cache": cache}, expected, schema)


class DsaIndexerPrepareReference(_IndexKCacheReference):
    def reference(
        self,
        *,
        x,
        q_lora,
        wq_b,
        wk,
        weights_proj,
        k_norm_weight,
        k_norm_bias,
        cos_cache,
        sin_cache,
        positions,
        out_cache_loc,
        index_k_cache,
        k_norm_eps,
        **_,
    ):
        """Each query head bf16(q_lora wq_b^T), turned and transformed as the key, over its
        scale; head_weights bf16(bf16(x weights_proj^T) * n_heads^-0.5) * that scale *
        HEAD_DIM^-0.5; the keys stored at their slots, and the slots, for check."""
        n_heads = weights_proj.shape[0]
        cos, sin = _rotary(positions, cos_cache, sin_cache)
        keys, key_scales = _keys(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, cos, sin)
        q = _project(q_lora.float(), wq_b.float()).reshape(-1, n_heads, HEAD_DIM)
        q = _rotate(q, cos[:, None], sin[:, None])
        q_scales = _scales(q)
        gate = _bf16(_project(x.float(), weights_proj.float()) * n_heads**-0.5)
        return {
            "q_fp8": (q / q_scales[..., None]).to(torch.float8_e4m3fn),
            "head_weights": gate * q_scales * HEAD_DIM**-0.5,
            "index_k_cache": _stored_cache(index_k_cache, out_cache_loc, keys, key_scales),
            "written_slots": out_cache_loc,
        }


class DsaIndexerKeyPrepareReference(_IndexKCacheReference):
    def reference(
        self,
        *,
        x,
        wk,
        k_norm_weight,
        k_norm_bias,
        cos_cache,
        sin_cache,
        positions,
        out_cache_loc,
        index_k_cache,
        k_norm_eps,
        **_,
    ):
        """Nothing, the keys stored at their slots, and the slots, for check."""
        cos, sin = _rotary(positions, cos_cache, sin_cache)
        keys, key_scales = _keys(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, cos, sin)
        cache = _stored_cache(index_k_cache, out_cache_loc, keys, key_scales)
        return {
            "no_values": torch.zeros(0, device=cache.device),
            "index_k_cache": cache,
            "written_slots": out_cache_loc,
        }
