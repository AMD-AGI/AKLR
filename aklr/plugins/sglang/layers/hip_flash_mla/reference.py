# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

__all__ = ["deepseek_v4_sparse_mla_decode_inv_rope_reference"]

if TYPE_CHECKING:
    from torch import Tensor

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
