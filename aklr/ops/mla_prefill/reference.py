# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent double-precision CSR attention with a denominator-only sink."""


def _decode_nope(raw):
    import torch

    shape = raw.shape[:-1]
    codes = raw[..., :448].contiguous().view(torch.float8_e4m3fn).double()
    scales = torch.exp2(raw[..., 448:462].double() - 127)
    return (codes.reshape(*shape, 14, 32) * scales.unsqueeze(-1)).reshape(*shape, 448)


def sparse_prefill_mxfp8_reference(point, row_major_args, to_float64):
    import torch

    args = row_major_args
    query = torch.cat((_decode_nope(args["q_nope"]), args["q_rope"].double()), dim=-1)
    result = torch.zeros_like(query, dtype=torch.float64)
    # Decode only referenced rows. Huge prefix pools need not be materialized
    # in float64 just to evaluate the sparse reference.
    for start in range(0, query.shape[0], 16):
        stop = min(start + 16, query.shape[0])
        selected, masks = [], []
        for pool in ("prefix", "extend"):
            offsets = args[f"{pool}_indptr"][start : stop + 1].long()
            lengths = offsets[1:] - offsets[:-1]
            width = int(lengths.max())
            columns = torch.arange(width, device=offsets.device)
            masks.append(columns.unsqueeze(0) < lengths.unsqueeze(1))
            positions = offsets[:-1].unsqueeze(1) + columns.unsqueeze(0)
            if width:
                positions = positions.clamp_max(args[f"{pool}_indices"].numel() - 1)
            ids = args[f"{pool}_indices"][positions].long()
            selected.append(
                torch.cat(
                    (_decode_nope(args[f"{pool}_nope"][ids]), args[f"{pool}_rope"][ids].double()), dim=-1
                )
            )
        values = torch.cat(selected, dim=1)
        valid = torch.cat(masks, dim=1)
        logits = torch.bmm(query[start:stop], values.transpose(1, 2)) * args["softmax_scale"]
        logits = logits.masked_fill(~valid.unsqueeze(1), -torch.inf)
        sink = args["sinks"].double().view(1, -1, 1).expand(stop - start, -1, -1)
        joined = torch.cat((logits, sink), dim=-1)
        probabilities = torch.softmax(joined, dim=-1)[..., :-1]
        result[start:stop] = torch.bmm(probabilities, values)
    return result


_QUERY_ROWS = 128  # queries a step: [heads, 128, keys] float32 logits stay under 1 GiB at 131072 keys


def _operand(x, point):
    """x as the kernels attend over it: with qkv_fp8 rounded to float8_e4m3fn, as aiter's fp8 prefill rounds it."""
    import torch

    return x.to(torch.float8_e4m3fn).to(x.dtype) if point.attrs["qkv_fp8"] else x


def mla_prefill_varlen_reference(point, row_major_args, decode):
    """Each head's causal attention per sequence, queries aligned at the end of its keys, in the precision decode gives."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    # in the precision decode gives (float64 on a retry): finite bf16 products can overflow float32
    q, k, v = (_operand(decode(args[name], dtypes[name]), point) for name in ("q", "k", "v"))
    starts_q = args["cu_seqlens_q"].tolist()
    cu_k = args.get("cu_seqlens_k")
    starts_k = starts_q if cu_k is None else cu_k.tolist()
    output = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
    for s in range(len(starts_q) - 1):
        q0, q1, k0, k1 = starts_q[s], starts_q[s + 1], starts_k[s], starts_k[s + 1]
        n_q, n_k = q1 - q0, k1 - k0
        keys, values = k[k0:k1], v[k0:k1]
        for r0 in range(0, n_q, _QUERY_ROWS):
            r1 = min(r0 + _QUERY_ROWS, n_q)
            logits = torch.einsum("qhd,khd->hqk", q[q0 + r0 : q0 + r1], keys) * args["sm_scale"]
            last = torch.arange(r0, r1, device=q.device)[:, None] + (n_k - n_q)
            logits = logits.masked_fill(torch.arange(n_k, device=q.device)[None] > last, -torch.inf)
            output[q0 + r0 : q0 + r1] = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), values)
    return output


def mla_prefill_varlen_lse_reference(point, row_major_args, decode):
    """Each head's attention per sequence, causal (queries aligned at the end of its keys) or over every key of the
    sequence, in the precision decode gives: the output and the natural log-sum-exp of the logits read, [query row, head]."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    causal = point.attrs["causal"]
    # in the precision decode gives (float64 on a retry): finite bf16 products can overflow float32
    q, k, v = (_operand(decode(args[name], dtypes[name]), point) for name in ("q", "k", "v"))
    starts_q = args["cu_seqlens_q"].tolist()
    cu_k = args.get("cu_seqlens_k")
    starts_k = starts_q if cu_k is None else cu_k.tolist()
    output = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
    lse = q.new_zeros(q.shape[0], q.shape[1])
    for s in range(len(starts_q) - 1):
        q0, q1, k0, k1 = starts_q[s], starts_q[s + 1], starts_k[s], starts_k[s + 1]
        n_q, n_k = q1 - q0, k1 - k0
        keys, values = k[k0:k1], v[k0:k1]
        for r0 in range(0, n_q, _QUERY_ROWS):
            r1 = min(r0 + _QUERY_ROWS, n_q)
            logits = torch.einsum("qhd,khd->hqk", q[q0 + r0 : q0 + r1], keys) * args["sm_scale"]
            if causal:
                last = torch.arange(r0, r1, device=q.device)[:, None] + (n_k - n_q)
                logits = logits.masked_fill(torch.arange(n_k, device=q.device)[None] > last, -torch.inf)
            output[q0 + r0 : q0 + r1] = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), values)
            lse[q0 + r0 : q0 + r1] = logits.logsumexp(dim=-1).transpose(0, 1)
    return output, lse
