# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["muse_spark_layer_reference"]

_HEAD_DIM = 64
_Q_LEN = 8
_TOP_K = 8
_RMS_EPS = 1e-5
_POST_EPS = 1e-8
_ROUTE_EPS = 1e-15
_ROUTER_MULT = 0.17677669529663687
_QK_SCALE = 5.4730064863875
_FP8_MAX = 448.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _bf16(x: Tensor) -> Tensor:
    import torch

    return x.to(torch.bfloat16).to(x.dtype)


def _rms(x: Tensor, eps: float) -> Tensor:
    import torch

    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _mxfp4(x: Tensor) -> Tensor:
    """x through aiter's MXFP4 activation quantization and back: per 32 values, the E8M0
    scale of the block maximum rounded at mantissa bit 21, E2M1 rounded half to even and
    saturated; computed on float32 bits, the values are exact in any wider float."""
    import torch

    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    exponent = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    scaled = blocks * torch.ldexp(torch.ones_like(amax), -exponent)
    a = scaled.abs()
    ab = a.view(torch.int32)
    subnormal = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    normal = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(a >= 6.0, torch.full_like(subnormal, 7), torch.where(a < 1.0, subnormal, normal))
    magnitude = torch.tensor(_E2M1, device=x.device)[code.long()]
    value = torch.where(scaled < 0, -magnitude, magnitude)
    return (value * torch.ldexp(torch.ones_like(amax), exponent)).flatten(-2).to(x.dtype)


def _mx_weight(args: Mapping[str, Any], name: str, expert: int, decode: Callable) -> Tensor:
    w = decode(args[name][expert], "float4_e2m1")
    scale = decode(args[f"{name}_scale"][expert], "float8_e8m0")
    return w * scale.repeat_interleave(32, dim=-1)


def _attention(args: Mapping[str, Any], window: int, scale: float, decode: Callable) -> Tensor:
    """One rank's heads: each row over its request's keys, causal among the step's rows,
    the probabilities rounded to bf16; query head h reads KV head h // (heads / KV heads)."""
    import torch

    q = decode(args["q"], "bfloat16")
    k_cache = decode(args["k_cache"], "float8_e4m3fn")
    v_cache = decode(args["v_cache"], "float8_e4m3fn")
    table, lens = args["page_table"].long(), args["context_lens"].long()
    rows = q.shape[0]
    out = torch.empty_like(q)
    for r in range(rows // _Q_LEN):
        context = int(lens[r])
        keys = context + _Q_LEN
        pages = table[r, : (keys + 15) // 16]
        slots = (pages[:, None] * 16 + torch.arange(16, device=pages.device)).reshape(-1)[:keys]
        k, v = k_cache[slots.to(k_cache.device)], v_cache[slots.to(v_cache.device)]
        position = torch.arange(keys, device=q.device)
        for row in range(_Q_LEN):
            t = r * _Q_LEN + row
            visible = position <= context + row
            if window:
                visible &= context + row - position < window
            grouped = q[t].unflatten(0, (k.shape[1], -1))  # [KV heads, heads a KV head, 64]
            logits = scale * torch.einsum("gid,kgd->gik", grouped, k)
            p = torch.softmax(logits.masked_fill(~visible, float("-inf")), dim=-1)
            out[t] = torch.einsum("gik,kgd->gid", _bf16(p), v).flatten(0, 1)
    return _bf16(out)


def _gated(o: Tensor, gate: Tensor) -> Tensor:
    import torch

    return _bf16(_rms(o, _RMS_EPS) * torch.sigmoid(gate)).flatten(1)


def _route(x: Tensor, args: Mapping[str, Any], decode: Callable) -> tuple[Tensor, Tensor]:
    """The top-8 of sigmoid + bias (a tie to the lower id) and their normalized sigmoids."""
    import torch

    logits = x @ decode(args["router_hi"], "bfloat16").T + x @ decode(args["router_lo"], "bfloat16").T
    sigmoid = torch.sigmoid(logits * _ROUTER_MULT)
    biased = sigmoid + decode(args["router_bias"], "float32")
    ids = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :_TOP_K]
    picks = sigmoid.gather(1, ids)
    return ids, picks / (picks.sum(-1, keepdim=True) + _ROUTE_EPS)


def _experts(
    pooled: Tensor,
    ids: Tensor,
    scores: Tensor,
    group_args: Sequence[Mapping[str, Any]],
    decode: Callable,
    inter_dim: int,
) -> Tensor:
    """Each pick's down(silu(gate) * up) on the experts the ranks' slices make whole, MXFP4
    activations, times the expert norm's gain, RMS-normalized, weighted, summed."""
    import torch
    import torch.nn.functional as F

    device = pooled.device
    gain = decode(group_args[0]["expert_norm_weight"], "bfloat16")
    x = _mxfp4(pooled)
    out = torch.zeros_like(pooled)
    for expert in ids.unique().tolist():
        # rank r holds the r-th slice of w13's gate rows and of its up rows, of w2's rows
        halves = [_mx_weight(a, "w13", expert, decode).to(device).chunk(2) for a in group_args]
        w13 = torch.cat([h[0] for h in halves] + [h[1] for h in halves])
        w2 = torch.cat([_mx_weight(a, "w2", expert, decode).to(device) for a in group_args])
        tokens, slot = (ids == expert).nonzero(as_tuple=True)
        gate, up = (x[tokens] @ w13.T).split(inter_dim, dim=-1)
        y = _bf16(_mxfp4(_bf16(F.silu(gate) * up)) @ w2.T)
        v = _bf16(y * gain)
        weight = scores[tokens, slot, None] * torch.rsqrt(v.square().mean(-1, keepdim=True) + _POST_EPS)
        out.index_add_(0, tokens, weight * v)
    return _bf16(out)


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Interleaved RoPE: pair (2i, 2i + 1) rotated by angle i, each product rounded."""
    import torch

    pairs = x.unflatten(-1, (-1, 2))
    other = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return _bf16(_bf16(x * cos) + _bf16(other * sin))


def muse_spark_layer_reference(
    point: Any,
    group_args: Sequence[Mapping[str, Any]],
    rank: int,
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, ...]:
    """The layer for ``rank``: next_input, next_k, next_v, then q, o_gate, residual,
    next_k_cache and next_v_cache as the call leaves them.

    Rounding points (bf16 unless noted), as the model's ops: the attention probabilities
    and output; the gated output; each rank's o_proj partial, then their fp32 sum; the
    weightless boundary norm; the pre-FFN norm; the pooled row and its norm; the experts'
    input and gated product in MXFP4, each expert output, it times its gain; the summed
    expert output; the unpooled row and it times post_ffn_weight; that normed; the next
    input; the next projection; q and k normed, q scaled; cos, sin and each RoPE product
    and their sum; the caches' rows in fp8 at +-448.
    """
    import torch

    args = group_args[rank]
    device = args["residual"].device
    window, scale = int(args["sliding_window"]), float(args["scale"])

    def shared(name: str, dtype: str) -> Tensor:
        return decode(args[name], dtype)

    # attention and o_proj, summed over the ranks in rank order
    projected = torch.zeros_like(shared("residual", "float32"))
    for a in group_args:
        gated = _gated(_attention(a, window, scale, decode), decode(a["o_gate"], "bfloat16"))
        projected += _bf16(gated.to(device) @ decode(a["o_proj"], "bfloat16").to(device).T)
    branch = _bf16(_rms(_bf16(projected), _POST_EPS))
    residual = shared("alpha", "float32") * shared("residual", "float32") + shared("beta", "float32") * branch
    x = _bf16(_rms(residual, _RMS_EPS) * shared("pre_norm_weight", "bfloat16"))

    # router, pooler and experts
    ids, scores = _route(x, args, decode)
    pooled = _bf16(x @ shared("pooler", "bfloat16").T)
    pooled = _bf16(_rms(pooled, _RMS_EPS) * shared("pooled_norm_weight", "bfloat16"))
    expert_sum = _experts(pooled, ids, scores, group_args, decode, point.axes["inter_dim"])
    unpooled = _bf16(expert_sum @ shared("unpooler", "bfloat16").T)
    branch = _bf16(_rms(_bf16(unpooled * shared("post_ffn_weight", "bfloat16")), _POST_EPS))
    residual = shared("alpha2", "float32") * residual + shared("beta2", "float32") * branch
    next_input = _rms(residual, _RMS_EPS) * shared("next_norm_weight", "bfloat16")

    # the next layer's projection, norms, RoPE and cache write
    projected = _bf16(_bf16(next_input) @ shared("next_qkvg", "bfloat16").T)
    q_width, kv_width = point.axes["n_heads"] * _HEAD_DIM, point.axes["n_kv_heads"] * _HEAD_DIM
    q, k, v, o_gate = (
        t.unflatten(-1, (-1, _HEAD_DIM))
        for t in projected.split((q_width, kv_width, kv_width, q_width), dim=-1)
    )
    q = _bf16(_bf16(_rms(q, _RMS_EPS)) * _QK_SCALE)
    k = _bf16(_rms(k, _RMS_EPS))
    if bool(args["next_rope"]):
        angles = shared("rotary", "float32")[args["positions"].long()]
        half = torch.arange(_HEAD_DIM, device=device) // 2
        cos, sin = _bf16(angles[:, None, half]), _bf16(angles[:, None, _HEAD_DIM // 2 + half])
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
    slots = args["slots"].long()
    if bool(args["next_sliding"]):
        slots = args["swa_map"].long()[slots]
    k_cache = shared("next_k_cache", "float8_e4m3fn").clone()
    v_cache = shared("next_v_cache", "float8_e4m3fn").clone()
    k_cache[slots] = k.clamp(-_FP8_MAX, _FP8_MAX)
    v_cache[slots] = v.clamp(-_FP8_MAX, _FP8_MAX)
    return next_input, k, v, q, o_gate, residual, k_cache, v_cache
