# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in float32: the inputs decoded exactly, nothing rounded
in between but what sglang's fp8 absorb quantizes, the activation it multiplies by the fp8
weights (_group_fp8). RoPE, the absorbed products and the qkv latent stay unrounded (the
kernels' bf16 steps are theirs, left to the bounds)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mla_qkv_a_norm_reference"]


def _rms_normalized(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x times the reciprocal rms of its row, plus eps under the root, times weight."""
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * (mean_square + eps).rsqrt() * weight


def mla_qkv_a_norm_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """(q_latent, kv_latent, k_rope): each token's hidden state times w_qkv_a, its q and KV
    parts RMS-normalized and scaled by their weights, its RoPE part as projected.
    Rounding points: none (sglang's bf16 qkv latent, before the norms, is left to the bounds)."""
    args = row_major_args

    def get(name: str) -> Tensor:
        return to_float32(args[name], point.dtypes[name])

    q_rank, kv_rank = point.axes["q_lora_rank"], point.axes["kv_lora_rank"]
    latent = get("hidden_states") @ get("w_qkv_a").T
    eps = args["eps"]
    return (
        _rms_normalized(latent[:, :q_rank], get("q_norm_weight"), eps),
        _rms_normalized(latent[:, q_rank : q_rank + kv_rank], get("kv_norm_weight"), eps),
        latent[:, q_rank + kv_rank :],
    )
