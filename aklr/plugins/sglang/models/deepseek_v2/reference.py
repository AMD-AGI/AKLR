# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in fp32: every input decoded to fp32 and nothing
rounded or quantized in between (the kernels' bf16, fp8 and fp4 steps are theirs) but
what the checkpoints' quantization schemes quantize: an fp8 linear's input (_linear),
and what deepseek_v2_moe's contract rounds: the router's logits, in the dtype sglang's
MoEGate returns them in. The index-K cache holds a key as its format does, fp8 values at
a power-of-2 scale; the values the reference returns there are unrounded."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from aklr.ops.topk.reference import topk_transform_paged_reference

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "deepseek_v2_attention_indexer_reference",
    "deepseek_v2_attention_reference",
    "deepseek_v2_dense_layer_reference",
    "deepseek_v2_eh_proj_reference",
    "deepseek_v2_fp8_attention_indexer_reference",
    "deepseek_v2_fp8_dense_layer_reference",
    "deepseek_v2_fp8_moe_layer_reference",
    "deepseek_v2_fp8_moe_layer_shared_apart_reference",
    "deepseek_v2_moe_layer_reference",
    "deepseek_v2_moe_reference",
    "deepseek_v2_unquantized_moe_layer_reference",
    "deepseek_v2_unquantized_moe_layer_shared_apart_reference",
]

_MXFP4_BLOCK = 32
_PAGE = 64  # keys per index-K cache page


def _fp32(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Callable[..., Tensor | None]:
    """Tensor argument ``name`` (its ``index`` only, if given) in fp32; None if
    absent."""

    def get(name: str, index: Any = None) -> Tensor | None:
        tensor = row_major_args.get(name)
        if tensor is None:
            return None
        if index is not None:
            tensor = tensor[index]
        return to_float64(tensor, point.dtypes[name]).float()

    return get


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    import torch

    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def _mlp(x: Tensor, gate_up: Tensor, down: Tensor) -> Tensor:
    """down(silu(gate) * up), gate and up the halves of x gate_up^T."""
    import torch

    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


def _linear(x: Tensor, weight: Tensor, scale: Tensor | None) -> Tensor:
    """x weight^T. An fp8 weight [n, k] comes with its scale per 128 x 128 block and takes
    x as the checkpoint's FP8 scheme quantizes it (transformers' FP8Linear, sglang's fp8
    linears): in bf16, as the model holds its activations, then each 1 x 128 group in fp8
    at a scale of its amax / 448."""
    import torch

    if scale is None:
        return x @ weight.T
    n, k = weight.shape
    block_scale = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    weight = weight * block_scale
    fp8 = torch.float8_e4m3fn
    groups = x.bfloat16().float().unflatten(-1, (-1, 128))
    group_scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    group_scale = group_scale.clamp(min=torch.finfo(torch.float32).tiny)
    x = ((groups / group_scale).to(fp8).float() * group_scale).flatten(-2)
    return x @ weight.T


def _fp8_mlp(x: Tensor, gate_up: Tensor, gate_up_scale: Tensor, down: Tensor, down_scale: Tensor) -> Tensor:
    """_mlp of fp8 linears (_linear)."""
    import torch

    gate, up = _linear(x, gate_up, gate_up_scale).chunk(2, dim=-1)
    return _linear(torch.nn.functional.silu(gate) * up, down, down_scale)


def _routed(
    point: Any,
    row_major_args: Mapping[str, Any],
    x: Tensor,
    router: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Each token's topk experts by sigmoid score plus correction_bias, and their
    weights: the renormalized scores times routed_scaling_factor."""
    return _routed_on(point, row_major_args, x @ router.T, bias)


def _routed_on(
    point: Any,
    row_major_args: Mapping[str, Any],
    logits: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """_routed on the router's ``logits``."""
    scores = logits.sigmoid()
    ids = (scores + bias).topk(point.attrs["topk"], dim=-1).indices
    weights = scores.gather(-1, ids)
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * row_major_args["routed_scaling_factor"]


def _experts(
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    mlp: Callable[[Tensor, int], Tensor],
) -> Tensor:
    """Each token's experts, ``mlp(rows, e)`` expert e on its rows, weighed and summed."""
    import torch

    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        out.index_add_(0, tokens, weights[tokens, slots, None] * mlp(x[tokens], e))
    return out


def _with_shared_slot(point: Any, ids: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """The routing with the shared expert appended: the last slot, at weight 1."""
    import torch

    shared = torch.full_like(ids[:, :1], point.axes["n_routed"])
    return (
        torch.cat([ids, shared], dim=-1),
        torch.cat([weights, torch.ones_like(weights[:, :1])], dim=-1),
    )


def deepseek_v2_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """On hidden_states, each token's routed experts and the shared one (the last slot) at
    weight 1, the MXFP4 weights dequantized per 32 values: deepseek_v2_moe_layer's experts
    without its norm and stream. Rounding points: the router's logits, exact products of
    hidden_states and router_weight summed in float64, round to hidden_states' dtype."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    hidden = args["hidden_states"]
    logits = (
        to_float64(hidden, point.dtypes["hidden_states"])
        @ to_float64(args["router_weight"], point.dtypes["router_weight"]).T
    )
    x = f("hidden_states")
    ids, weights = _routed_on(point, args, logits.to(hidden.dtype).float(), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, weight("w1", e), weight("w2", e))

    return _experts(x, ids, weights, mlp).double()


def deepseek_v2_moe_layer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus, on its post_attention_layernorm, each token's routed experts and
    the shared one (the last slot) at weight 1, the MXFP4 weights dequantized per 32
    values."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    ids, weights = _routed(point, args, x, f("router_weight"), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, weight("w1", e), weight("w2", e))

    return (stream + _experts(x, ids, weights, mlp)).double()


def deepseek_v2_fp8_moe_layer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus, on its post_attention_layernorm, each token's routed experts and
    the shared one (the last slot) at weight 1, each expert's two linears fp8 ones
    (_fp8_mlp)."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    ids, weights = _routed(point, args, x, f("router_weight"), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _fp8_mlp(rows, f("w1", e), f("w1_scale", e), f("w2", e), f("w2_scale", e))

    return (stream + _experts(x, ids, weights, mlp)).double()


def deepseek_v2_fp8_moe_layer_shared_apart_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus, on its post_attention_layernorm, each token's routed experts and
    the shared expert at weight 1, every one of their linears an fp8 one (_fp8_mlp)."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    ids, weights = _routed(point, args, x, f("router_weight"), f("correction_bias"))

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _fp8_mlp(rows, f("w1", e), f("w1_scale", e), f("w2", e), f("w2_scale", e))

    shared = _fp8_mlp(
        x,
        f("shared_gate_up_weight"),
        f("shared_gate_up_scale"),
        f("shared_down_weight"),
        f("shared_down_scale"),
    )
    return (stream + _experts(x, ids, weights, mlp) + shared).double()


def deepseek_v2_unquantized_moe_layer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus, on its post_attention_layernorm, each token's routed experts and
    the shared one (the last slot) at weight 1."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    ids, weights = _routed(point, args, x, f("router_weight"), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, f("w1", e), f("w2", e))

    return (stream + _experts(x, ids, weights, mlp)).double()


def deepseek_v2_unquantized_moe_layer_shared_apart_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus, on its post_attention_layernorm, each token's routed experts and
    the shared expert at weight 1."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    ids, weights = _routed(point, args, x, f("router_weight"), f("correction_bias"))

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, f("w1", e), f("w2", e))

    routed = _experts(x, ids, weights, mlp)
    shared = _mlp(x, f("shared_gate_up_weight"), f("shared_down_weight"))
    return (stream + routed + shared).double()


def deepseek_v2_dense_layer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus DeepseekV2MLP's output on its post_attention_layernorm."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    return (stream + _mlp(x, f("gate_up_weight"), f("down_weight"))).double()


def deepseek_v2_fp8_dense_layer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The stream plus DeepseekV2MLP's output on its post_attention_layernorm, its two
    linears fp8 ones (_fp8_mlp)."""
    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    stream = f("hidden_states")
    x = _rms_norm(stream, f("norm_weight"), args["norm_eps"])
    mlp = _fp8_mlp(x, f("gate_up_weight"), f("gate_up_scale"), f("down_weight"), f("down_scale"))
    return (stream + mlp).double()


def deepseek_v2_eh_proj_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """eh_proj of both normed halves."""
    import torch

    args, f = row_major_args, _fp32(point, row_major_args, to_float64)
    embeds = _rms_norm(f("embeds"), f("enorm_weight"), args["eps"])
    hidden = _rms_norm(f("hidden"), f("hnorm_weight"), args["eps"])
    return (torch.cat([embeds, hidden], dim=-1) @ f("weight").T).double()


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x turned pair by pair, pair i being (x[2i], x[2i + 1])."""
    import torch

    out = torch.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin
    return out


def _hadamard(x: Tensor) -> Tensor:
    """x times the Sylvester Hadamard matrix over its last dim, a power of 2
    (fast_hadamard_transform's)."""
    import torch

    n, h, y = x.shape[-1], 1, x
    while h < n:
        y = y.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
        h *= 2
    return y.reshape(x.shape)


def _mla_qkv(
    point: Any, row_major_args: Mapping[str, Any], f: Callable[..., Tensor | None]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """x, the normed stream; qkv_a, then the RMSNorms and q_b: x, q_lora, q, k_latent,
    k_pe."""
    args, axes = row_major_args, point.axes
    x = _rms_norm(f("hidden_states"), f("input_norm_weight"), args["input_norm_eps"])
    qkv = _linear(x, f("qkv_a_weight"), f("qkv_a_scale"))
    q_lora_rank, kv_lora_rank = axes["q_lora_rank"], axes["kv_lora_rank"]
    rope_dim = qkv.shape[-1] - q_lora_rank - kv_lora_rank
    q_a, kv, k_pe = qkv.split([q_lora_rank, kv_lora_rank, rope_dim], dim=-1)
    q_lora = _rms_norm(q_a, f("q_a_norm_weight"), args["eps"])
    k_latent = _rms_norm(kv, f("kv_a_norm_weight"), args["eps"])
    q = _linear(q_lora, f("q_b_weight"), f("q_b_scale"))
    return x, q_lora, q, k_latent, k_pe


def _mla_sparse(
    row_major_args: Mapping[str, Any],
    f: Callable[..., Tensor | None],
    q: Tensor,
    k_latent: Tensor,
    k_pe: Tensor,
    topk_slots: Tensor,
) -> tuple[Tensor, Tensor]:
    """Each token's KV row stored into the cache at its slot, then each head's attention
    over the rows its top-k slots name, through W_UV and o_proj, plus the stream: the
    output, then the cache."""
    import torch

    args = row_major_args
    w_uk_t, w_uv = f("w_uk_t"), f("w_uv")
    w_scale = 1.0 if args.get("w_scale") is None else f("w_scale").reshape(())
    n_heads, latent, nope = w_uk_t.shape
    q = q.view(q.shape[0], n_heads, -1)
    positions = args["positions"].long()
    cos, sin = f("cos_cache")[positions], f("sin_cache")[positions]
    cache = f("kv_cache")
    rows = torch.cat([k_latent, _rope(k_pe, cos, sin)], dim=-1)
    cache[args["out_cache_loc"].long()] = rows
    slots = topk_slots.long()
    keys = cache[slots.clamp(min=0)]
    q_latent = torch.einsum("thd,hrd->thr", q[..., :nope], w_uk_t) * w_scale
    q_pe = _rope(q[..., nope:], cos[:, None], sin[:, None])
    scores = torch.einsum("thc,tkc->thk", torch.cat([q_latent, q_pe], -1), keys)
    scores = (scores * args["sm_scale"]).masked_fill((slots < 0)[:, None], -torch.inf)
    # a token that reads no row gets 0
    weights = scores.softmax(-1).nan_to_num(0.0)
    o_latent = torch.einsum("thk,tkr->thr", weights, keys[..., :latent])
    o = torch.einsum("thr,hvr->thv", o_latent, w_uv) * w_scale
    output = _linear(o.flatten(1), f("o_proj_weight"), f("o_proj_scale"))
    return f("hidden_states") + output, cache


def deepseek_v2_attention_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """_mla_qkv on the stream, then _mla_sparse over topk_indices: the output, then the
    cache."""
    f = _fp32(point, row_major_args, to_float64)
    _, _, q, k_latent, k_pe = _mla_qkv(point, row_major_args, f)
    topk = row_major_args["topk_indices"]
    output, cache = _mla_sparse(row_major_args, f, q, k_latent, k_pe, topk)
    return output.double(), cache.double()


def _tiles(pages: Tensor, ix_dim: int) -> Tensor:
    """The index-K cache pages' fp8 values, 64 keys' a page, as [page, key // 16,
    value // 16, key % 16, value % 16]; their fp32 scales follow them in the page."""
    return pages[:, : _PAGE * ix_dim].view(-1, _PAGE // 16, ix_dim // 16, 16, 16)


def _index_keys(cache: Tensor, ix_dim: int) -> Tensor:
    """The index-K cache's keys by slot, each its fp8 values times its scale."""
    import torch

    raw = cache.view(torch.uint8)
    values = _tiles(raw, ix_dim).permute(0, 1, 3, 2, 4).reshape(-1, ix_dim)
    values = values.contiguous().view(torch.float8_e4m3fn).float()
    scales = raw[:, _PAGE * ix_dim :].contiguous().view(torch.float32).reshape(-1, 1)
    return values * scales


def deepseek_v2_attention_indexer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """_mla_qkv on the stream; the indexer's q, k and heads' weights; each token's key
    stored into the index-K cache; each token's scores over its positions and their
    top-k slots (topk_transform_paged's); then _mla_sparse over those slots. The output,
    the top-k, then both caches."""
    import torch

    args, axes = row_major_args, point.axes
    f = _fp32(point, args, to_float64)
    x, q_lora, q, k_latent, k_pe = _mla_qkv(point, args, f)
    n_tokens, ix_heads, ix_dim = q.shape[0], axes["ix_heads"], axes["ix_dim"]
    rope_dim = 2 * args["index_cos_cache"].shape[-1]
    positions = args["positions"].long()
    cos, sin = f("index_cos_cache")[positions], f("index_sin_cache")[positions]

    def turned(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """RoPE on t's first rope_dim values, then the Hadamard transform times
        ix_dim^-0.5."""
        t = torch.cat([_rope(t[..., :rope_dim], cos, sin), t[..., rope_dim:]], -1)
        return _hadamard(t) * ix_dim**-0.5

    index_q = _linear(q_lora, f("wq_b"), f("wq_b_scale"))
    index_q = index_q.view(n_tokens, ix_heads, ix_dim)
    index_q = turned(index_q, cos[:, None], sin[:, None])
    index_k = _linear(x, f("wk"), f("wk_scale"))
    index_k = torch.nn.functional.layer_norm(
        index_k, (ix_dim,), f("k_norm_weight"), f("k_norm_bias"), args["k_norm_eps"]
    )
    index_k = turned(index_k, cos, sin)
    head_weights = (x @ f("weights_proj").T) * ix_heads**-0.5 * ix_dim**-0.5

    loc = args["out_cache_loc"].long()
    keys = _index_keys(args["index_k_cache"], ix_dim)
    keys[loc] = index_k
    table = args["page_table"].long()
    scores = torch.zeros(n_tokens, _PAGE * table.shape[1], device=x.device)
    for m, length in enumerate(args["seq_lens"].tolist()):
        t = torch.arange(length, device=x.device)
        slots = table[m, t // _PAGE] * _PAGE + t % _PAGE
        relu = (index_q[m] @ keys[slots].T).clamp(min=0)
        scores[m, :length] = head_weights[m] @ relu
    topk_indices = topk_transform_paged_reference(
        SimpleNamespace(axes=axes, attrs={"page_size": _PAGE}),
        {"scores": scores, "seq_lens": args["seq_lens"], "page_tables": table},
        to_float64,
    )
    output, kv_cache = _mla_sparse(args, f, q, k_latent, k_pe, topk_indices.int())

    # the tokens' keys as the cache's format holds them: fp8 values at the power of 2 at
    # or above amax / 448, the values unrounded
    amax = index_k.abs().amax(-1).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    stored = args["index_k_cache"].clone()
    page, key = loc // _PAGE, loc % _PAGE
    raw = stored.view(torch.uint8)
    scale_bytes = scale.view(torch.uint8).view(n_tokens, 4)
    raw[:, _PAGE * ix_dim :].view(-1, _PAGE, 4)[page, key] = scale_bytes
    index_k_cache = to_float64(stored, point.dtypes["index_k_cache"])
    values = (index_k / scale[:, None]).double().view(n_tokens, ix_dim // 16, 16)
    _tiles(index_k_cache, ix_dim)[page, key // 16, :, key % 16] = values
    return output.double(), topk_indices, index_k_cache, kv_cache.double()


def deepseek_v2_fp8_attention_indexer_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """deepseek_v2_attention_indexer_reference, every linear fp8 (_linear)."""
    return deepseek_v2_attention_indexer_reference(point, row_major_args, to_float64)
