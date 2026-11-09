# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["deepseek_v4_dp_moe_gather_baseline"]


def deepseek_v4_dp_moe_gather_baseline(hidden_states, num_token_non_padded, out_dtype=None, out=None):
    """What sglang 0.5.23's MoeRowTransport runs on ROCm before aiter's custom all-gather of a
    DP-attention rank's rows: _mask_dp_pad_rows (arange, >=, masked_fill_ in place), then the
    copy of the rows into aiter's registered input buffer that all_gather_unreg makes, here into
    the caller's out."""
    import torch
    from sglang.srt.models.deepseek_common.moe_transport import _mask_dp_pad_rows

    if out is None:
        out = torch.empty_like(hidden_states)
    _mask_dp_pad_rows(hidden_states, _PaddedRows(num_token_non_padded))
    out.copy_(hidden_states)
    return out


class _PaddedRows:
    """The one forward_batch field _mask_dp_pad_rows reads."""

    def __init__(self, num_token_non_padded):
        self.num_token_non_padded = num_token_non_padded
