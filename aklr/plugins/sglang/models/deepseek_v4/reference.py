# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

__all__ = [
    "deepseek_v4_attention_q_prep_reference",
    "deepseek_v4_sparse_mla_decode_inv_rope_reference",
    "deepseek_v4_wo_a_split_k_mxfp8_reference",
    "deepseek_v4_wo_a_split_k_reference",
]


if TYPE_CHECKING:
    from torch import Tensor


def deepseek_v4_attention_q_prep_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(q, q_lora_quant, q_lora_scale, kv_cache): the queries roped, wq_b's operand and its
    scales, and the pool with each token's KV row written at its slot.

    Rounding points of the op. Query side: the normalized q_lora is rounded to BF16; its
    per-32 scale exponent is that of max(amax, 1e-10) * float32(1 / 448) (a float32 product)
    rounded up, clamped to [1, 254]; the codes are E4M3FN (round to nearest even) of the
    clamped quotient; the GEMM's result is rounded to BF16 before its RoPE. The mean square,
    the GEMM's sums and the query RoPE are exact here, their float32 roundings left to compare.
    KV side, all float32 and bit for bit, since the pool bytes compare exactly: the sum of
    squares adds the exact squares of each pair (2t, 2t + 1), then the 256 pair sums as
    xor-butterflies over 8 groups of 32 (strides 16, 8, 4, 2, 1), then the 8 group sums
    (strides 4, 2, 1), each step one float32 add; nf = 1 / sqrt(sum / 512 + eps), the add,
    the square root and the quotient each rounded; a dim is (x * nf) * w; a group of 64 takes
    the exponent of max(amax, 1e-4) / 448 rounded up and E4M3FN codes of the clamped scaled
    values; the RoPE pair (r, i) is fma(r, cos, -(i * sin)), fma(r, sin, i * cos), each
    product rounded and each fma rounded once, then rounded to BF16.
    """
    import torch

    rank, n_heads = point.axes["q_lora_rank"], point.axes["n_heads"]
    page_size = point.attrs["page_size"]
    qkv = decode(row_major_args["qkv_a"], point.dtypes["qkv_a"]).float()
    freqs = decode(row_major_args["freqs"], point.dtypes["freqs"]).float()
    positions = row_major_args["positions"].long()
    n_tokens, device = qkv.shape[0], qkv.device
    cos, sin = freqs[positions, 0::2], freqs[positions, 1::2]

    # query side: q_norm, its E4M3FN operand, wq_b, RoPE
    q_lora = qkv[:, :rank].double()
    q_norm_weight = decode(row_major_args["q_norm_weight"], point.dtypes["q_norm_weight"]).double()
    rstd = torch.rsqrt(q_lora.square().mean(-1, keepdim=True) + row_major_args["q_eps"])
    normed = (q_lora * rstd * q_norm_weight).to(torch.bfloat16).float().view(n_tokens, -1, 32)
    amax = normed.abs().amax(-1).clamp_min(1e-10)
    exponent = _exponent_up(amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=device))
    q_lora_scale = _power_of_two(exponent.clamp(1, 254))
    codes = (normed / q_lora_scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    q_lora_quant = codes.view(n_tokens, rank)
    wq_b = decode(row_major_args["wq_b"], point.dtypes["wq_b"]).double()
    wq_b_scale = decode(row_major_args["wq_b_scale"], point.dtypes["wq_b_scale"]).double()
    weight = wq_b * wq_b_scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    operand = (codes * q_lora_scale[..., None]).view(n_tokens, rank).double()
    q = (operand @ weight.t()).to(torch.bfloat16).double().view(n_tokens, n_heads, 512)
    real, imag = q[..., 448::2], q[..., 449::2]
    c, s = cos.double()[:, None], sin.double()[:, None]
    q[..., 448::2], q[..., 449::2] = real * c - imag * s, real * s + imag * c

    # KV side: kv_norm in the K store's order, its E4M3FN and BF16 parts, into the pool
    kv = qkv[:, rank:]
    pair_sums = (kv * kv).view(n_tokens, 256, 2)
    total = pair_sums[..., 0] + pair_sums[..., 1]
    groups = total.view(n_tokens, 8, 32)
    while groups.shape[-1] > 1:
        half = groups.shape[-1] // 2
        groups = groups[..., :half] + groups[..., half:]
    total = groups.view(n_tokens, 8)
    while total.shape[-1] > 1:
        half = total.shape[-1] // 2
        total = total[..., :half] + total[..., half:]
    total = total[:, 0]
    kv_eps = torch.tensor(row_major_args["kv_eps"], dtype=torch.float32, device=device)
    nf = 1.0 / torch.sqrt(total * (1.0 / 512.0) + kv_eps)
    kv_norm_weight = decode(row_major_args["kv_norm_weight"], point.dtypes["kv_norm_weight"]).float()
    normed_kv = kv * nf[:, None] * kv_norm_weight
    nope = normed_kv[:, :448].reshape(n_tokens, 7, 64)
    floor = torch.tensor(1e-4, dtype=torch.float32, device=device)
    kv_exponent = _exponent_up(torch.maximum(nope.abs().amax(-1), floor) / 448.0)
    kv_codes = (nope * _power_of_two(254 - kv_exponent)[..., None]).clamp(-448.0, 448.0)
    real, imag = normed_kv[:, 448::2], normed_kv[:, 449::2]
    rope = torch.stack((_fma32(real, cos, -(imag * sin)), _fma32(real, sin, imag * cos)), -1)
    data = torch.cat(
        (
            kv_codes.to(torch.float8_e4m3fn).view(torch.uint8).view(n_tokens, 448),
            rope.view(n_tokens, 64).to(torch.bfloat16).view(torch.uint8),
        ),
        -1,
    )

    kv_cache = row_major_args["kv_cache"].clone()
    loc = row_major_args["swa_loc"].long()
    rows = loc >= 0
    page, slot = loc[rows] // page_size, loc[rows] % page_size
    data_at = (page * kv_cache.shape[1] + slot * 576)[:, None] + torch.arange(576, device=device)
    scale_at = (page * kv_cache.shape[1] + 576 * page_size + slot * 8)[:, None] + torch.arange(
        7, device=device
    )
    flat = kv_cache.view(-1)
    flat[data_at.reshape(-1)] = data[rows].reshape(-1)
    flat[scale_at.reshape(-1)] = kv_exponent[rows].to(torch.uint8).reshape(-1)
    return q, q_lora_quant.double(), q_lora_scale.double(), kv_cache.double()


def _exponent_up(x: Tensor) -> Tensor:
    """The biased exponent of each positive float32 in x, plus 1 unless x is a power of two."""
    import torch

    bits = x.float().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()


def _power_of_two(exponent: Tensor) -> Tensor:
    """The float32 2 ** (exponent - 127) of biased exponents in [1, 254]."""
    import torch

    return (exponent.int() << 23).view(torch.float32)


def _fma32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """fma(a, b, c) of float32 tensors rounded once to float32: a * b is exact in float64; the
    float64 sum is rounded again only when it lands on a float32 tie, which its error breaks."""
    import torch

    product, addend = a.double() * b.double(), c.double()
    total = product + addend
    shifted = total - product
    error = (product - (total - shifted)) + (addend - shifted)
    rounded = total.float()
    toward = torch.where(total > rounded.double(), float("inf"), float("-inf")).float()
    neighbour = torch.nextafter(rounded, toward).double()
    is_tie = (total != rounded.double()) & (total == (rounded.double() + neighbour) / 2)
    nudged = torch.nextafter(total, total + error)
    return torch.where(is_tie & (error != 0), nudged, total).float()


# Gathered KV values per chunk of query rows: 256 MiB of float32.
_CHUNK_ELEMENTS = 2**26


def deepseek_v4_sparse_mla_decode_inv_rope_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """output: each head's softmax over q . row * sm_scale of the rows its query row reads
    (negative slots never weigh), its sink in the normalisation only, weighing the decoded
    512-wide rows; 0 for a row that reads no slot. Then the last 64 lanes, per (cos, sin)
    pair of rope_freqs[position], become even * cos + odd * sin and odd * cos - even * sin.

    A pool row decodes exactly: NoPE value j is its fp8 e4m3 code times 2 ** (e - 127), e the
    exponent byte of group j // 64; the RoPE values are bf16. Nothing is rounded in between:
    the kernels' bf16 weights and dot operands, and the output's bf16 rounding before the
    rotation (sglang stores it, then its rope pass reloads it), are error left to compare.
    """
    import torch

    args = row_major_args
    q = decode(args["q"], point.dtypes["q"])
    n_tokens, n_heads, head_dim = q.shape
    pools = [(args["kv_cache"], args["sparse_indices"].long())]
    if point.attrs["has_extra_kv_cache"]:
        pools.append((args["extra_kv_cache"], args["extra_sparse_indices"].long()))
    for cache, slots in pools:
        if (slots >= cache.shape[0] * cache.shape[1]).any():
            raise ValueError("a sparse index is outside its KV pool")
    visible = torch.cat([slots >= 0 for _, slots in pools], dim=-1)

    def rows_of(chunk: slice) -> Tensor:
        """The decoded rows a chunk of query rows reads, [rows, slots, 512]; skipped slots
        read slot 0, hidden by ``visible``."""
        rows = []
        for cache, slots in pools:
            page_size = cache.shape[1]
            raw = cache.view(torch.uint8).flatten()
            picked = slots[chunk].clamp(min=0)
            within = (picked % page_size)[..., None]
            page = (picked // page_size)[..., None] * (page_size * 584)
            payload = page + within * 576
            columns = torch.arange(448, device=raw.device)
            codes = raw[payload + columns].view(torch.float8_e4m3fn).float()
            exponents = raw[page + page_size * 576 + within * 8 + columns[:7]].float() - 127
            nope = codes * torch.exp2(exponents).repeat_interleave(64, dim=-1)
            rope = raw[payload + 448 + columns[:128]].view(torch.bfloat16).float()
            rows.append(torch.cat((nope, rope), dim=-1))
        return torch.cat(rows, dim=1).to(q.dtype)

    sinks = decode(args["sinks"], point.dtypes["sinks"])
    output = q.new_zeros(n_tokens, n_heads, head_dim)
    n_rows = max(1, _CHUNK_ELEMENTS // (visible.shape[-1] * head_dim))
    for start in range(0, n_tokens, n_rows):
        chunk = slice(start, start + n_rows)
        rows = rows_of(chunk)
        logits = torch.einsum("thd,tkd->thk", q[chunk], rows) * args["sm_scale"]
        logits = logits.masked_fill(~visible[chunk][:, None], -torch.inf)
        logits = torch.cat((logits, sinks[:, None].expand(*logits.shape[:2], 1)), dim=-1)
        weights = logits.softmax(dim=-1)[..., :-1]
        output[chunk] = weights @ rows

    freqs = decode(args["rope_freqs"], point.dtypes["rope_freqs"])[args["positions"].long()]
    cos, sin = freqs[:, None, 0::2], freqs[:, None, 1::2]
    rope_dim = freqs.shape[-1]
    even, odd = output[..., -rope_dim::2], output[..., -rope_dim + 1 :: 2]
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    return torch.cat((output[..., :-rope_dim], rotated.flatten(-2)), dim=-1)


def _wo_a_bf16(
    point: Any, row_major_args: Mapping[str, Any], decode: Callable[[Tensor, str], Tensor]
) -> Tensor:
    """x @ w.T per group, exact in float64 (in_dim bf16 products), rounded to bf16 once; float64
    [n_tokens, out_dim]."""
    import torch

    x = decode(row_major_args["x"], point.dtypes["x"]).double()
    w = decode(row_major_args["w"], point.dtypes["w"]).double()
    y = torch.einsum("tgd,grd->tgr", x, w).reshape(x.shape[0], -1)
    return y.to(torch.bfloat16).double()


def _fp8_grid(y: Tensor) -> tuple[Tensor, Tensor]:
    """(y / 2**e rounded to e4m3, e + 127) per 32 columns of y, by sglang's fp8_grid_quant: amax
    floored at 1e-10, times float32 1/448 rounded to float32, e the exponent of the smallest power
    of two at or above it (clamped to the e8m0 range), y / 2**e saturated at 448, then e4m3
    round-to-nearest-even; both float64."""
    import torch

    groups = y.reshape(y.shape[0], -1, 32)
    floor = float(torch.tensor(1e-10, dtype=torch.float32))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    raw = (groups.abs().amax(-1).clamp_min(floor) * divisor).to(torch.float32).contiguous().view(torch.int32)
    biased = (((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254).double()
    scaled = (groups / torch.exp2(biased - 127).unsqueeze(-1)).clamp(-448.0, 448.0)
    return scaled.to(torch.float8_e4m3fn).double().reshape(y.shape), biased


def deepseek_v4_wo_a_split_k_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """out: x[:, g] @ w[g].T for every group, side by side, and with fp8_grid on wo_b's fp8 grid.

    Rounding points of the op: the exact sum rounds to bf16 once; with fp8_grid, that bf16 value
    is quantized by sglang's fp8_grid_quant (``_fp8_grid``) and dequantized exactly. The
    float32 accumulation (sglang sums eight K-split partials in split order) is error left to
    compare.
    """
    import torch

    y = _wo_a_bf16(point, row_major_args, decode)
    if not point.attrs["fp8_grid"]:
        return y
    q, biased = _fp8_grid(y)
    scale = torch.exp2(biased - 127).repeat_interleave(32, dim=-1)
    return q * scale


def deepseek_v4_wo_a_split_k_mxfp8_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(quantized, scales): x[:, g] @ w[g].T for every group, side by side, as wo_b's MXFP8
    operand.

    Rounding points of the op: the exact sum rounds to bf16 once, then sglang's fp8_grid_quant
    (``_fp8_grid``) gives the e4m3 codes and each 32-column group's scale 2**e. The float32
    accumulation is error left to compare.
    """
    import torch

    q, biased = _fp8_grid(_wo_a_bf16(point, row_major_args, decode))
    return q, torch.exp2(biased - 127)
