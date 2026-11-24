# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent CSR attention with a denominator-only sink."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["MlaPrefillVarlenLseReference", "MlaPrefillVarlenReference", "SparsePrefillMxfp8Reference"]


def _decode_nope(raw):
    shape = raw.shape[:-1]
    codes = raw[..., :448].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(raw[..., 448:462].float() - 127)
    return (codes.reshape(*shape, 14, 32) * scales.unsqueeze(-1)).reshape(*shape, 448)


class SparsePrefillMxfp8Reference(OpReference):
    def reference(
        self,
        *,
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
        softmax_scale,
        **_,
    ):
        """Each query's attention over its prefix and extend rows, a sink in the softmax denominator."""
        query = torch.cat((_decode_nope(q_nope), q_rope.float()), dim=-1)
        pools = (
            (prefix_indptr, prefix_indices, prefix_nope, prefix_rope),
            (extend_indptr, extend_indices, extend_nope, extend_rope),
        )
        result = torch.zeros_like(query)
        # Decode only referenced rows: huge prefix pools need not be materialized.
        for start in range(0, query.shape[0], 16):
            stop = min(start + 16, query.shape[0])
            selected, masks = [], []
            for indptr, indices, nope, rope in pools:
                offsets = indptr[start : stop + 1].long()
                lengths = offsets[1:] - offsets[:-1]
                width = int(lengths.max())
                columns = torch.arange(width, device=offsets.device)
                masks.append(columns.unsqueeze(0) < lengths.unsqueeze(1))
                positions = offsets[:-1].unsqueeze(1) + columns.unsqueeze(0)
                if width:
                    positions = positions.clamp_max(indices.numel() - 1)
                ids = indices[positions].long()
                selected.append(torch.cat((_decode_nope(nope[ids]), rope[ids].float()), dim=-1))
            values = torch.cat(selected, dim=1)
            valid = torch.cat(masks, dim=1)
            logits = torch.bmm(query[start:stop], values.transpose(1, 2)) * softmax_scale
            logits = logits.masked_fill(~valid.unsqueeze(1), -torch.inf)
            sink = sinks.float().view(1, -1, 1).expand(stop - start, -1, -1)
            joined = torch.cat((logits, sink), dim=-1)
            probabilities = torch.softmax(joined, dim=-1)[..., :-1]
            result[start:stop] = torch.bmm(probabilities, values)
        return {"output": result.to(torch.bfloat16)}


_QUERY_ROWS = 128  # queries a step: [heads, 128, keys] float32 logits stay under 1 GiB at 131072 keys


def _operands(q, k, v, qkv_fp8):
    """q, k and v in float32 as the kernels attend over them: with qkv_fp8 rounded to float8_e4m3fn, as aiter's fp8
    prefill rounds them."""
    q, k, v = q.float(), k.float(), v.float()
    if qkv_fp8:
        q, k, v = (x.to(torch.float8_e4m3fn).to(x.dtype) for x in (q, k, v))
    return q, k, v


class MlaPrefillVarlenReference(OpReference):
    def reference(self, *, q, k, v, cu_seqlens_q, sm_scale, qkv_fp8, cu_seqlens_k=None, **_):
        """Each head's causal attention per sequence, queries aligned at the end of its keys."""
        dtype, (q, k, v) = q.dtype, _operands(q, k, v, qkv_fp8)
        starts_q = cu_seqlens_q.tolist()
        starts_k = starts_q if cu_seqlens_k is None else cu_seqlens_k.tolist()
        output = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
        for s in range(len(starts_q) - 1):
            q0, q1, k0, k1 = starts_q[s], starts_q[s + 1], starts_k[s], starts_k[s + 1]
            n_q, n_k = q1 - q0, k1 - k0
            keys, values = k[k0:k1], v[k0:k1]
            for r0 in range(0, n_q, _QUERY_ROWS):
                r1 = min(r0 + _QUERY_ROWS, n_q)
                logits = torch.einsum("qhd,khd->hqk", q[q0 + r0 : q0 + r1], keys) * sm_scale
                last = torch.arange(r0, r1, device=q.device)[:, None] + (n_k - n_q)
                logits = logits.masked_fill(torch.arange(n_k, device=q.device)[None] > last, -torch.inf)
                output[q0 + r0 : q0 + r1] = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), values)
        return {"output": output.to(dtype)}


class MlaPrefillVarlenLseReference(OpReference):
    def reference(self, *, q, k, v, cu_seqlens_q, sm_scale, qkv_fp8, causal, cu_seqlens_k=None, **_):
        """Each head's attention per sequence, causal (queries aligned at the end of its keys) or over every key of the
        sequence: the output and the natural log-sum-exp of the logits read, [query row, head]."""
        dtype, (q, k, v) = q.dtype, _operands(q, k, v, qkv_fp8)
        starts_q = cu_seqlens_q.tolist()
        starts_k = starts_q if cu_seqlens_k is None else cu_seqlens_k.tolist()
        output = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
        lse = q.new_zeros(q.shape[0], q.shape[1])
        for s in range(len(starts_q) - 1):
            q0, q1, k0, k1 = starts_q[s], starts_q[s + 1], starts_k[s], starts_k[s + 1]
            n_q, n_k = q1 - q0, k1 - k0
            keys, values = k[k0:k1], v[k0:k1]
            for r0 in range(0, n_q, _QUERY_ROWS):
                r1 = min(r0 + _QUERY_ROWS, n_q)
                logits = torch.einsum("qhd,khd->hqk", q[q0 + r0 : q0 + r1], keys) * sm_scale
                if causal:
                    last = torch.arange(r0, r1, device=q.device)[:, None] + (n_k - n_q)
                    logits = logits.masked_fill(torch.arange(n_k, device=q.device)[None] > last, -torch.inf)
                output[q0 + r0 : q0 + r1] = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), values)
                lse[q0 + r0 : q0 + r1] = logits.logsumexp(dim=-1).transpose(0, 1)
        return {"output": output.to(dtype), "lse": lse}
