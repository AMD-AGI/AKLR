# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch: the packed pools decoded exactly, nothing rounded in
between."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16
from aklr.ops.mla.reference import flash_mla_with_kvcache_reference
from aklr.plugins.sglang.models.deepseekv41.contract import (
    GROUP_HEADS,
    HEAD_DIM,
    KV_SCALE_BLOCK,
    MXFP8_BLOCK,
    ROPE_DIM,
)

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "deepseekv41_mono_k1_reference",
    "deepseekv41_mono_k2_attention_reference",
    "deepseekv41_mono_k2_ffn_reference",
    "deepseekv41_mxfp8_linear_reference",
    "deepseekv41_sparse_mla_reference",
]


def deepseekv41_mxfp8_linear_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Rounding point: a bf16 x rounds to fp8 e4m3 at its scale per 1 x 32 before the product."""
    import torch

    args = row_major_args
    x = decode(args["x"], point.dtypes["x"])
    blocks = x.unflatten(-1, (-1, MXFP8_BLOCK))
    if point.attrs["has_x_scale"]:
        x_scale = decode(args["x_scale"], point.dtypes["x_scale"]).to(x.dtype)[..., None]
    else:
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        mantissa, exponent = torch.frexp(amax / 448.0)
        # ceil(log2): frexp's exponent, one less at a power of two
        x_scale = torch.exp2((exponent - (mantissa == 0.5).int()).clamp(-127, 127).to(x.dtype))
        blocks = (blocks / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)
    weight = decode(args["weight"], point.dtypes["weight"]).to(x.dtype)
    weight_scale = decode(args["weight_scale"], point.dtypes["weight_scale"]).to(x.dtype)
    weight_scale = weight_scale.repeat_interleave(MXFP8_BLOCK, dim=0).repeat_interleave(MXFP8_BLOCK, dim=1)
    return (blocks * x_scale).flatten(-2) @ (weight * weight_scale).mT


def deepseekv41_sparse_mla_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """flash_mla_with_kvcache's sparse attention over the same pools, then the rotation."""
    import torch

    args = row_major_args
    # the sparse branch of flash_mla_with_kvcache reads these of its Point, and no lengths
    sparse = SimpleNamespace(attrs={"kv_cache_format": "dsv4_fp8"}, dtypes=point.dtypes)
    output, _ = flash_mla_with_kvcache_reference(
        sparse,
        {
            "q": args["q"],
            "kv_cache": args["kv_cache"],
            "sparse_indices": args["indices"],
            "extra_kv_cache": args.get("extra_kv_cache"),
            "extra_sparse_indices": args.get("extra_indices"),
            "sinks": args["sinks"],
            "sm_scale": args["sm_scale"],
            "kv_scale": 1.0,
        },
        decode,
    )
    if not point.attrs["has_rope_freqs"]:
        return output
    freqs = decode(args["rope_freqs"], point.dtypes["rope_freqs"]).to(output.dtype)
    freqs = freqs[args["positions"].long()].view(-1, 1, 1, ROPE_DIM // 2, 2)
    cos, sin = freqs.unbind(-1)
    rope = output[..., -ROPE_DIM:].unflatten(-1, (ROPE_DIM // 2, 2))
    a, b = rope.unbind(-1)
    rope.copy_(torch.stack((a * cos + b * sin, b * cos - a * sin), dim=-1))
    return output


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    import torch

    # the mean square in float64: a float32 square overflows on large finite inputs
    return x * torch.rsqrt(x.double().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def _rotated(values: Tensor, freqs: Tensor) -> Tensor:
    """``values``' pairs (a, b) rotated by ``freqs``' (c, s), broadcast over the dims between."""
    import torch

    a, b = values.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


def _mxfp8_linear(x: Tensor, args: Mapping[str, Any], point: Any, name: str, decode: Callable) -> Tensor:
    """deepseekv41_mxfp8_linear of the fp32 rows ``x`` on the weight ``name``."""
    linear = SimpleNamespace(
        attrs={"has_x_scale": False},
        dtypes={"x": "float32", "weight": point.dtypes[name], "weight_scale": point.dtypes[f"{name}_scale"]},
    )
    arguments = {"x": x, "weight": args[name], "weight_scale": args[f"{name}_scale"]}
    return deepseekv41_mxfp8_linear_reference(
        linear, arguments, lambda tensor, dtype: tensor if dtype == "float32" else decode(tensor, dtype)
    )


def _boundary(
    point: Any,
    args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
    x: Tensor | None,
    pre_mix: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The mHC boundary of the contracts: (the normed layer input, the streams, and this
    boundary's pre, post and combination mixes). Rounding points: the streams and the layer
    input round to bf16; the mixes read the rounded streams."""
    import torch

    def f(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name]).float()

    streams = f("residual")
    if x is not None:
        streams = x[:, None, :] * f("post_mix")[:, :, None] + f("comb_mix").mT @ streams
        streams = rounded_to_bfloat16(streams).float()
    if point.dtypes.get(pre_mix) is None:
        layer_input = streams[:, 0]
    else:
        layer_input = rounded_to_bfloat16((streams * f(pre_mix)[:, :, None]).sum(dim=1)).float()
    flat = streams.flatten(1)
    rms = torch.rsqrt(flat.double().square().mean(dim=-1, keepdim=True) + args["hc_rms_eps"]).float()
    pre, post, comb = ((flat @ f("hc_fn").mT) * rms).split((4, 4, 16), dim=-1)
    scale, base, eps = f("hc_scale"), f("hc_base"), args["hc_eps"]
    pre = torch.sigmoid(pre * scale[0] + base[:4]) + eps
    post = 2.0 * torch.sigmoid(post * scale[1] + base[4:8])
    comb = (comb * scale[2] + base[8:]).unflatten(-1, (4, 4)).softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(args["sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    normed = _rms_norm(layer_input, f("norm_weight"), args["norm_eps"])
    return normed, streams, pre, post, comb


def deepseekv41_mono_k1_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, ...]:
    """The outputs in the contract's order. Rounding points, past the boundary's: the normed
    input, the qkv latent, q's normed latent and q round to bf16; each MXFP8 linear rounds its
    input to fp8; the record's position-free part rounds to fp8 at its scales."""
    import torch

    args = row_major_args

    def f(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name]).float()

    x = f("x") if point.attrs["has_x"] else None
    normed, streams, pre, post, comb = _boundary(point, args, decode, x, "prev_pre_mix")
    normed = rounded_to_bfloat16(normed).float()
    qkv = rounded_to_bfloat16(_mxfp8_linear(normed, args, point, "wqkv", decode)).float()
    q_latent, kv = qkv.split((qkv.shape[-1] - HEAD_DIM, HEAD_DIM), dim=-1)
    q_latent = rounded_to_bfloat16(_rms_norm(q_latent, f("q_norm_weight"), args["norm_eps"])).float()
    freqs = f("rope_freqs")[args["positions"].long()]
    q = rounded_to_bfloat16(_mxfp8_linear(q_latent, args, point, "wq_b", decode)).float()
    q = q.unflatten(-1, (-1, HEAD_DIM))
    q = torch.cat((q[..., :-ROPE_DIM], _rotated(q[..., -ROPE_DIM:], freqs[:, None, :])), dim=-1)
    kv = _rms_norm(kv, f("kv_norm_weight"), args["norm_eps"])
    k_rope = _rotated(kv[:, -ROPE_DIM:], freqs)
    blocks = kv[:, :-ROPE_DIM].unflatten(-1, (-1, KV_SCALE_BLOCK))
    mantissa, exponent = torch.frexp(blocks.abs().amax(dim=-1).clamp(min=1e-4) / 448.0)
    exponent = exponent - (mantissa == 0.5).int()
    k_nope = (blocks * torch.exp2(-exponent.float())[..., None]).clamp(-448.0, 448.0).flatten(1)
    k_nope = k_nope.to(torch.float8_e4m3fn).float()
    return q, streams, pre, post, comb, normed, q_latent, k_nope, (exponent + 127).float(), k_rope


def deepseekv41_mono_k2_attention_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, ...]:
    """The outputs in the contract's order. Rounding points, past the boundary's: the attended
    values, wo_a's output and wo_b's round to bf16; wo_b rounds its input to fp8."""
    import torch

    args = row_major_args
    attended = rounded_to_bfloat16(deepseekv41_sparse_mla_reference(point, args, decode)).float()
    wo_a = decode(args["wo_a"], point.dtypes["wo_a"]).float()
    n_groups = point.axes["n_heads"] // GROUP_HEADS
    grouped = attended.reshape(attended.shape[0], n_groups, -1)
    projected = torch.einsum("tgd,grd->tgr", grouped, wo_a.unflatten(0, (n_groups, -1)))
    projected = rounded_to_bfloat16(projected).float().flatten(1)
    x = rounded_to_bfloat16(_mxfp8_linear(projected, args, point, "wo_b", decode)).float()
    return _boundary(point, args, decode, x, "pre_mix")


def _mxfp8(x: Tensor) -> Tensor:
    """``x`` in fp8 e4m3 at an e8m0 scale per 1 x 32, the scale 2**ceil(log2(amax / 448))."""
    import torch

    blocks = x.unflatten(-1, (-1, MXFP8_BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    mantissa, exponent = torch.frexp(amax / 448.0)
    scale = torch.exp2((exponent - (mantissa == 0.5).int()).clamp(-126, 127).to(x.dtype))
    return ((blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype) * scale).flatten(-2)


def deepseekv41_mono_k2_ffn_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token's routed experts and the shared one (the last slot) at weight 1. Rounding
    points: the router's logits, exact products summed in float64, round once to fp32;
    each expert linear's input goes to MXFP8 per 1 x 32, the clamped SwiGLU product to bf16
    first."""
    import torch
    import torch.nn.functional as F

    args = row_major_args

    def f(name: str, index: Any = None) -> Tensor:
        tensor = args[name] if index is None else args[name][index]
        return decode(tensor, point.dtypes[name]).float()

    hidden = f("hidden_states")
    scores = F.softplus((hidden.double() @ f("router_weight").double().T).float()).sqrt()
    ids = (scores + f("correction_bias")).topk(point.attrs["topk"], dim=-1).indices
    weights = scores.gather(-1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * args["routed_scaling_factor"]
    ids = torch.cat([ids, torch.full_like(ids[:, :1], point.axes["n_routed"])], dim=-1)
    weights = torch.cat([weights, torch.ones_like(weights[:, :1])], dim=-1)

    def weight(name: str, e: int) -> Tensor:
        return f(name, e) * f(f"{name}_scale", e).repeat_interleave(MXFP8_BLOCK, dim=-1)

    limit = args["swiglu_limit"]
    x = _mxfp8(hidden)
    output = torch.zeros_like(hidden)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        gate, up = (x[tokens] @ weight("w1", e).T).chunk(2, dim=-1)
        gated = rounded_to_bfloat16(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))
        output.index_add_(0, tokens, weights[tokens, slots, None] * (_mxfp8(gated) @ weight("w2", e).T))
    return output
