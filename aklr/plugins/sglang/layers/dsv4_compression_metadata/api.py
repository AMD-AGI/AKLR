# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsv4_compression_metadata.contract import DSV4_COMPRESSION_METADATA

__all__ = ["dsv4_compression_metadata"]


@api(DSV4_COMPRESSION_METADATA)
def dsv4_compression_metadata(
    seq_lens,
    positions,
    raw_out_loc,
    page_table,
    c4_out_loc,
    c4_positions,
    c4_seq_lens_raw,
    c4_seq_lens_clamp1,
    c128_out_loc,
    c128_positions,
    c128_seq_lens_raw,
    c128_seq_lens_clamp1,
    c128_page_indices,
): ...


@sglang_route(
    "sglang.kernels.ops.attention.dsv4.metadata_kernel.init_compression_metadata", versions=("0.5.23",)
)
def _init_compression_metadata(
    seq_lens, positions, raw_out_loc, page_table=None, page_size=0, compute_page_indices=True
):
    """The HIP DeepSeek-V4 backend's per-step compressed-cache metadata with its ratio-128 page indices, as
    one op call into the buffers sglang allocates; the call without page indices stays on sglang."""
    import torch

    if (
        not compute_page_indices
        or page_table is None
        or page_size < 128
        or page_size % 128
        or page_table.dim() != 2
    ):
        return FallbackReason.OUT_OF_CONTRACT
    rows, n_write = seq_lens.shape[0], raw_out_loc.shape[0]
    if n_write == 0 or n_write > rows or page_table.shape[0] != rows or page_table.dtype != torch.int32:
        return FallbackReason.OUT_OF_CONTRACT
    device = seq_lens.device
    width = (page_size // 128) * page_table.shape[1]

    def ints(n, dtype=torch.int32):
        return torch.empty(n, dtype=dtype, device=device)

    return dsv4_compression_metadata(
        seq_lens=seq_lens,
        positions=positions,
        raw_out_loc=raw_out_loc,
        page_table=page_table,
        c4_out_loc=ints(n_write, torch.int64),
        c4_positions=ints(rows),
        c4_seq_lens_raw=ints(rows),
        c4_seq_lens_clamp1=ints(rows),
        c128_out_loc=ints(n_write, torch.int64),
        c128_positions=ints(rows),
        c128_seq_lens_raw=ints(rows),
        c128_seq_lens_clamp1=ints(rows),
        c128_page_indices=torch.empty((rows, width), dtype=torch.int32, device=device),
    )
