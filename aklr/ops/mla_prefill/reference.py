# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["sparse_prefill_mxfp8_reference"]


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
