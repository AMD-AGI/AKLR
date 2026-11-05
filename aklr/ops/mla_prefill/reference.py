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
