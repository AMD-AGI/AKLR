# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "mla_prefill_varlen_lse_reference",
    "mla_prefill_varlen_reference",
    "sparse_prefill_mxfp8_reference",
]


def sparse_prefill_mxfp8_reference(
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    softmax_scale=512**-0.5,
) -> Tensor:
    """CSR attention over both pools in float64; a head's sink joins only the softmax's
    denominator."""
    import torch

    def decode_nope(raw):
        shape = raw.shape[:-1]
        codes = raw[..., :448].contiguous().view(torch.float8_e4m3fn).double()
        scales = torch.exp2(raw[..., 448:462].double() - 127)
        return (codes.reshape(*shape, 14, 32) * scales.unsqueeze(-1)).reshape(*shape, 448)

    pools = {
        "prefix": (prefix_nope, prefix_rope, prefix_indices, prefix_indptr),
        "extend": (extend_nope, extend_rope, extend_indices, extend_indptr),
    }
    query = torch.cat((decode_nope(q_nope), q_rope.double()), dim=-1)
    result = torch.zeros_like(query, dtype=torch.float64)
    # decode only the rows a query reads: a large prefix pool would not fit in float64
    for start in range(0, query.shape[0], 16):
        stop = min(start + 16, query.shape[0])
        selected, masks = [], []
        for nope, rope, indices, indptr in pools.values():
            offsets = indptr[start : stop + 1].long()
            lengths = offsets[1:] - offsets[:-1]
            width = int(lengths.max())
            columns = torch.arange(width, device=offsets.device)
            masks.append(columns.unsqueeze(0) < lengths.unsqueeze(1))
            positions = offsets[:-1].unsqueeze(1) + columns.unsqueeze(0)
            if width:
                positions = positions.clamp_max(indices.numel() - 1)
            ids = indices[positions].long()
            selected.append(torch.cat((decode_nope(nope[ids]), rope[ids].double()), dim=-1))
        values = torch.cat(selected, dim=1)
        valid = torch.cat(masks, dim=1)
        logits = torch.bmm(query[start:stop], values.transpose(1, 2)) * softmax_scale
        logits = logits.masked_fill(~valid.unsqueeze(1), -torch.inf)
        sink = sinks.double().view(1, -1, 1).expand(stop - start, -1, -1)
        probabilities = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]
        result[start:stop] = torch.bmm(probabilities, values)
    return result


def mla_prefill_varlen_lse_reference(
    q, k, v, cu_seqlens_q, qkv_fp8=True, causal=True, cu_seqlens_k=None, sm_scale=None
) -> tuple[Tensor, Tensor]:
    """Each head's attention per sequence, causal (queries aligned at the end of its keys) or over every key of
    the sequence, in float32: the output and the natural log-sum-exp of the logits read, [query row, head]. With
    qkv_fp8 over q, k and v rounded to float8_e4m3fn, as aiter's fp8 prefill rounds them."""
    import torch

    if qkv_fp8:
        q, k, v = (x.to(torch.float8_e4m3fn) for x in (q, k, v))
    q, k, v = q.float(), k.float(), v.float()
    starts_q = cu_seqlens_q.tolist()
    starts_k = starts_q if cu_seqlens_k is None else cu_seqlens_k.tolist()
    output = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
    lse = q.new_zeros(q.shape[0], q.shape[1])
    for s in range(len(starts_q) - 1):
        q0, q1, k0, k1 = starts_q[s], starts_q[s + 1], starts_k[s], starts_k[s + 1]
        n_q, n_k = q1 - q0, k1 - k0
        keys, values = k[k0:k1], v[k0:k1]
        # 128 queries a step: [heads, 128, keys] float32 logits stay under 1 GiB at 131072 keys
        for r0 in range(0, n_q, 128):
            r1 = min(r0 + 128, n_q)
            logits = torch.einsum("qhd,khd->hqk", q[q0 + r0 : q0 + r1], keys) * sm_scale
            if causal:
                last = torch.arange(r0, r1, device=q.device)[:, None] + (n_k - n_q)
                logits = logits.masked_fill(torch.arange(n_k, device=q.device)[None] > last, -torch.inf)
            output[q0 + r0 : q0 + r1] = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), values)
            lse[q0 + r0 : q0 + r1] = logits.logsumexp(dim=-1).transpose(0, 1)
    return output, lse


def mla_prefill_varlen_reference(
    q, k, v, cu_seqlens_q, qkv_fp8=True, cu_seqlens_k=None, sm_scale=None
) -> Tensor:
    """Each head's causal attention per sequence, queries aligned at the end of its keys."""
    return mla_prefill_varlen_lse_reference(q, k, v, cu_seqlens_q, qkv_fp8, True, cu_seqlens_k, sm_scale)[0]
