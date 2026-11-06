# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["moe_sort_mxfp8_quant_reference"]


def _scale_byte_offsets(rows: Tensor, n_groups: int, scale_cols: int) -> Tensor:
    """aiter's mx_scale_shuffle_idx: the byte of (sorted row, group) in sorted_scales, [rows, groups]."""
    import torch

    r = rows[:, None]
    g = torch.arange(n_groups, device=rows.device)[None, :]
    return (
        (r // 32) * (32 * scale_cols)
        + (g // 8) * 256
        + (g % 4) * 64
        + (r % 16) * 4
        + (g % 8) // 4 * 2
        + (r % 32) // 16
    )


def moe_sort_mxfp8_quant_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, ...]:
    """(num_valid_ids, moe_buf, quantized, sorted_ids, sorted_weights, sorted_expert_ids,
    sorted_scales), the integers and the copied buffers in float64.

    Rounding points of the op: the scale amax * (1/448) rounds to float32 (amax floored at
    float32's 1e-10), then its exponent rounds up; the division by that power of two is exact,
    and the quotient, at most 448 in magnitude, rounds to E4M3FN. The sort is exact.
    """
    import torch

    block, n_tokens = point.attrs["block_size"], point.axes["n_tokens"]
    topk, n_experts = point.axes["topk"], point.attrs["n_experts"]
    n_groups = point.axes["hidden_size"] // 32
    scale_cols = (n_groups + 7) // 8 * 8
    x = decode(row_major_args["hidden_states"], point.dtypes["hidden_states"]).float()
    weights = decode(row_major_args["topk_weights"], point.dtypes["topk_weights"]).double()
    device = x.device

    # pairs by expert, then by t * topk + k; each expert's rows in whole blocks
    experts = row_major_args["topk_ids"].reshape(-1).long()
    order = torch.sort(experts, stable=True).indices
    counts = torch.bincount(experts, minlength=n_experts)
    blocks = (counts + block - 1) // block
    first_block = torch.cumsum(blocks, 0) - blocks
    first_pair = torch.cumsum(counts, 0) - counts
    sorted_experts = experts[order]
    rank = torch.arange(order.numel(), device=device) - first_pair[sorted_experts]
    rows = first_block[sorted_experts] * block + rank
    n_blocks = int(blocks.sum())
    n_rows = n_blocks * block

    sorted_ids = row_major_args["sorted_ids"].double().clone()
    sorted_weights = decode(row_major_args["sorted_weights"], point.dtypes["sorted_weights"]).double().clone()
    sorted_ids[:n_rows] = float((topk << 24) | n_tokens)
    sorted_weights[:n_rows] = 0.0
    sorted_ids[rows] = ((order % topk) * 2**24 + order // topk).double()
    sorted_weights[rows] = weights.reshape(-1)[order]
    sorted_expert_ids = row_major_args["sorted_expert_ids"].double().clone()
    sorted_expert_ids[:n_blocks] = torch.repeat_interleave(
        torch.arange(n_experts, device=device), blocks
    ).double()
    num_valid_ids = torch.tensor([n_rows, n_tokens], dtype=torch.float64, device=device)

    groups = x.reshape(n_tokens, n_groups, 32)
    amax = groups.abs().amax(-1).clamp_min(float(torch.tensor(1e-10, dtype=torch.float32)))
    divisor = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))
    raw = (amax * divisor).to(torch.float32).contiguous().view(torch.int32)
    exponent = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).contiguous().view(torch.float32).double()
    values = (groups.double() / scale.unsqueeze(-1)).to(torch.float32).clamp(-448.0, 448.0)
    quantized = values.to(torch.float8_e4m3fn).double().reshape(n_tokens, n_groups * 32)
    quantized[(weights == 0).all(dim=-1)] = 0.0

    sorted_scales = row_major_args["sorted_scales"].double().clone().reshape(-1)
    pair_exponents = exponent.double()[order // topk]
    pair_exponents[weights.reshape(-1)[order] == 0] = 0.0
    sorted_scales[_scale_byte_offsets(rows, n_groups, scale_cols)] = pair_exponents
    sorted_scales = sorted_scales.reshape(-1, scale_cols)

    moe_buf = torch.zeros((n_tokens, n_groups * 32), dtype=torch.float64, device=device)
    return num_valid_ids, moe_buf, quantized, sorted_ids, sorted_weights, sorted_expert_ids, sorted_scales
