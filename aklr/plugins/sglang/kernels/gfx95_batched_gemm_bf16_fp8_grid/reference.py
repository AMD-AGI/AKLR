# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

__all__ = ["deepseek_v4_wo_a_split_k_mxfp8_reference", "deepseek_v4_wo_a_split_k_reference"]

if TYPE_CHECKING:
    from torch import Tensor


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
