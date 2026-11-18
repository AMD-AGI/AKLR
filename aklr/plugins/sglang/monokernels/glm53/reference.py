# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in the dtype ``decode`` returns: nothing rounded in between but
what a contract rounds. Each rank's layer runs on its own inputs, on its own device; the group's sums
run in rank order on the calling rank's."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16
from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM
from aklr.plugins.sglang.models.deepseek_v2.reference import (
    _experts,
    _first_largest,
    _fp8_linear,
    _with_shared_slot,
)

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["glm53_moe_layer_reference"]


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """GPT-J style: each pair 2i, 2i + 1 rotated by cos[i], sin[i]."""
    import torch

    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)


def _attention(point: Any, args: Mapping[str, Any], decode: Callable) -> tuple[Tensor, Tensor]:
    """This rank's attention output, before the group's sum, and its KV pool after the writes.
    Rounding point: each written row, cast to the pool's dtype."""
    import torch

    def get(name: str) -> Tensor:
        return decode(args[name], point.dtypes[name])

    ax, eps = point.axes, args["eps"]
    x = _rms_norm(get("hidden_states"), get("input_norm_weight"), eps)
    q_a, latent, k_pe = _fp8_linear(x, get("qkv_a_weight"), get("qkv_a_scale")).split(
        [ax["q_lora_rank"], ax["kv_lora_rank"], ROPE_DIM], dim=-1
    )
    q = _fp8_linear(_rms_norm(q_a, get("q_a_norm_weight"), eps), get("q_b_weight"), get("q_b_scale"))
    q_nope, q_pe = q.unflatten(-1, (ax["n_heads"], -1)).split([ax["qk_nope_dim"], ROPE_DIM], dim=-1)
    cos, sin = get("cos_sin_cache")[args["positions"]].chunk(2, dim=-1)
    q_pe, k_pe = _rope(q_pe, cos[:, None], sin[:, None]), _rope(k_pe, cos, sin)

    pool = args["kv_cache"].clone()
    rows = torch.cat([_rms_norm(latent, get("kv_a_norm_weight"), eps), k_pe], dim=-1)
    pool[args["cache_locations"]] = rows.to(pool.dtype)
    pool_values = decode(pool, point.dtypes["kv_cache"])

    w_scale = get("w_scale")
    slots = args["topk_indices"].long()
    read = pool_values[slots.clamp(min=0)]  # [tokens, topk, kv_dim]
    query = torch.cat([torch.einsum("thn,hnc->thc", q_nope, get("w_kc") * w_scale), q_pe], dim=-1)
    scores = torch.einsum("thc,tkc->thk", query, read) * args["sm_scale"]
    # a row that reads no slot gets 0
    weights = scores.masked_fill((slots < 0)[:, None], -torch.inf).softmax(-1).nan_to_num(0.0)
    latent_out = torch.einsum("thk,tkr->thr", weights, read[..., : ax["kv_lora_rank"]])
    heads = torch.einsum("thr,hrv->thv", latent_out, get("w_vc") * w_scale).flatten(1)
    return _fp8_linear(heads, get("o_weight"), get("o_scale")), pool_values


def _moe(point: Any, args: Mapping[str, Any], decode: Callable, stream: Tensor) -> Tensor:
    """This rank's MoE output on the stream, before the group's sum: deepseek_v2_fp8_moe's, ungrouped.
    Rounding point: the router's logits, exact products summed in float64, round once to bf16."""
    import torch

    def get(name: str, index: Any = None) -> Tensor:
        tensor = args[name] if index is None else args[name][index]
        return decode(tensor, point.dtypes[name])

    x = _rms_norm(stream, get("post_norm_weight"), args["eps"])
    logits = rounded_to_bfloat16(x.double() @ get("router_weight").double().T).to(x.dtype)
    scores = logits.sigmoid()
    ids = _first_largest(scores + get("correction_bias"), point.attrs["topk"])
    weights = scores.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True) * args["routed_scaling_factor"]
    ids, weights = _with_shared_slot(point, ids, weights)

    def mlp(rows: Tensor, e: int) -> Tensor:
        gate, up = _fp8_linear(rows, get("w1", e), get("w1_scale", e)).chunk(2, dim=-1)
        return _fp8_linear(torch.nn.functional.silu(gate) * up, get("w2", e), get("w2_scale", e))

    return _experts(x, ids, weights, mlp)


def glm53_moe_layer_reference(
    point: Any,
    group_args: Sequence[Mapping[str, Any]],
    rank: int,
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """``rank``'s stream plus the group's attention outputs summed, then plus the group's MoE
    outputs on each rank's stream summed; and ``rank``'s KV pool after its writes."""
    attentions = [_attention(point, args, decode) for args in group_args]
    device = attentions[rank][0].device
    attention = sum(a.to(device) for a, _ in attentions)
    streams = [
        decode(args["hidden_states"], point.dtypes["hidden_states"]) + attention.to(a.device)
        for args, (a, _) in zip(group_args, attentions)
    ]
    moe = sum(_moe(point, args, decode, s).to(device) for args, s in zip(group_args, streams))
    return streams[rank] + moe, attentions[rank][1]
