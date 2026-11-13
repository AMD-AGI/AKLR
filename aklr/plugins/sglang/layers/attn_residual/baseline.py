# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's ROCm aggregation kernel, ``attn_res_hip``, one launch per call."""

from __future__ import annotations

__all__ = ["add_attn_res_baseline", "attn_res_baseline"]


def attn_res_baseline(
    prefix,
    bank,
    score_weight,
    norm_weight,
    score_eps=1e-6,
    norm_eps=1e-6,
    n_valid_rows=8,
    write_bank=False,
    normalized=None,
):
    """sglang's kernel into ``normalized``; it writes bank row n_valid_rows with ``write_bank``."""
    import torch
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    if normalized is None:
        normalized = torch.empty_like(prefix)
    attn_res_hip(
        prefix,
        bank,
        score_weight,
        norm_weight,
        normalized,
        n_valid_rows,
        score_eps,
        norm_eps,
        write_prefix=write_bank,
    )
    return normalized


def add_attn_res_baseline(
    prefix,
    addend,
    bank,
    score_weight,
    norm_weight,
    score_eps=1e-6,
    norm_eps=1e-6,
    n_valid_rows=8,
    write_bank=False,
    normalized=None,
    prefix_out=None,
):
    """sglang's kernel with the pending addend folded in, into ``normalized`` and ``prefix_out``."""
    import torch
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    if normalized is None:
        normalized = torch.empty_like(prefix)
    if prefix_out is None:
        prefix_out = torch.empty_like(prefix)
    attn_res_hip(
        prefix,
        bank,
        score_weight,
        norm_weight,
        normalized,
        n_valid_rows,
        score_eps,
        norm_eps,
        addend=addend,
        prefix_out=prefix_out,
        write_prefix=write_bank,
    )
    return normalized, prefix_out
