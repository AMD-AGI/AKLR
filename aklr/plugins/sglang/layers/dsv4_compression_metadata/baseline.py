# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["Dsv4CompressionMetadataBaseline"]

_NAMES = (
    "c4_out_loc",
    "c4_positions",
    "c4_seq_lens_raw",
    "c4_seq_lens_clamp1",
    "c128_out_loc",
    "c128_positions",
    "c128_seq_lens_raw",
    "c128_seq_lens_clamp1",
    "c128_page_indices",
)


class Dsv4CompressionMetadataBaseline(OpBaseline):
    def baseline(self, *, seq_lens, positions, raw_out_loc, page_table, c128_page_indices, **_):
        """sglang 0.5.23's init_compression_metadata with its ratio-128 page indices, one program per row."""
        from sglang.kernels.ops.attention.dsv4.metadata_kernel import init_compression_metadata

        page_size = 128 * (c128_page_indices.shape[1] // page_table.shape[1])
        outputs = init_compression_metadata(
            seq_lens, positions, raw_out_loc, page_table, page_size, compute_page_indices=True
        )
        return dict(zip(_NAMES, outputs, strict=True))
