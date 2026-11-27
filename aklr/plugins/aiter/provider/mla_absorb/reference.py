# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in float32: the inputs decoded exactly, nothing rounded
in between but what sglang's fp8 absorb quantizes, the activation it multiplies by the fp8
weights (_absorbed). RoPE, the absorbed products and the qkv latent stay unrounded (the
kernels' bf16 steps are theirs, left to the bounds)."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["FP8_GROUP", "MlaKcRopeCacheReference", "MlaVcReference"]

# The width of the groups sglang's fp8 absorb quantizes its activations in (aiter's a8w8
# batched GEMM with group_size=128): a group's fp8 codes at a scale of max(amax, 1e-10) / 448.
FP8_GROUP = 128
# fp8 e4m3fn's largest finite value
_FP8_MAX = 448.0


def _absorbed(x, weight, scale):
    """x [tokens, heads, k] times weight [heads, n, k] and the per-tensor scale, x taken as
    sglang's fp8 absorb quantizes it (aiter's a8w8 batched GEMM, group_size=128): each
    128-wide group of k (the last one maybe narrower) in fp8 e4m3fn at a scale of
    max(amax, 1e-10) / 448, its codes x * (1 / scale) clamped to +-448, as the kernels compute
    them: 448 x / amax of bf16 values often falls on an fp8 tie, which another formula breaks
    the other way."""
    out = None
    for start in range(0, x.shape[-1], FP8_GROUP):
        group = x[..., start : start + FP8_GROUP]
        group_scale = group.abs().amax(-1, keepdim=True).clamp(min=1e-10) * (1.0 / _FP8_MAX)
        codes = (group * (1.0 / group_scale)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        part = torch.einsum("thk,hnk->thn", codes.float(), weight[..., start : start + FP8_GROUP])
        part = part * group_scale
        out = part if out is None else out + part
    return out * scale


def _rotated(x, cos, sin):
    """x's GPT-J pairs (2i, 2i + 1) rotated by the angles whose cos and sin are given, one
    per pair."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


class MlaKcRopeCacheReference(OpReference):
    def reference(
        self,
        *,
        q,
        qkv_latent,
        kv_latent,
        w_kc,
        w_scale,
        cos_cache,
        sin_cache,
        positions,
        out_cache_loc,
        kv_cache,
        **_,
    ):
        """Each head's position-free query part absorbed through w_kc, then its RoPE part
        rotated by the token's position; each token's KV latent and rotated RoPE key in its KV
        pool row. A row whose slot is 0 is padding: q_cat 0, no row written.
        Rounding points: the position-free query part's fp8 codes (_absorbed), and the outputs'."""
        nope, rope = w_kc.shape[-1], kv_cache.shape[-1] - kv_latent.shape[-1]
        q = q.float()
        positions = positions.long()
        cos, sin = cos_cache.float()[positions], sin_cache.float()[positions]
        q_cat = torch.cat(
            (
                _absorbed(q[..., :nope], w_kc.float(), w_scale.float()),
                _rotated(q[..., nope:], cos[:, None], sin[:, None]),
            ),
            -1,
        )
        slots = out_cache_loc.long()
        written = slots != 0
        q_cat = torch.where(written[:, None, None], q_cat, 0.0)
        rows = torch.cat((kv_latent.float(), _rotated(qkv_latent.float()[:, -rope:], cos, sin)), -1)
        dtype = kv_cache.dtype
        # a copy: .float() may return the very tensor it was given
        kv_cache = kv_cache.float().clone()
        kv_cache[slots[written]] = rows[written]
        return {"q_cat": q_cat.to(dtype), "kv_cache": kv_cache.to(dtype)}


class MlaVcReference(OpReference):
    def reference(self, *, attn_output, w_vc, w_scale, **_):
        """Each head's attention output absorbed through w_vc, the heads side by side.
        Rounding points: the attention output's fp8 codes (_absorbed), and the output's."""
        output = _absorbed(attn_output.float(), w_vc.float(), w_scale.float()).flatten(1)
        return {"attn_bmm_output": output.to(torch.bfloat16)}
