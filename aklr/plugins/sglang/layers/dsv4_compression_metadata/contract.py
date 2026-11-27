# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's per-step metadata of DeepSeek-V4's ratio-4 and ratio-128 compressed KV caches on HIP
(``init_compression_metadata`` in ``kernels/ops/attention/dsv4/metadata_kernel.py``, with its
ratio-128 page indices)."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, IntRangeDistribution, ToleranceBound
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.layers.dsv4_compression_metadata.baseline import Dsv4CompressionMetadataBaseline
from aklr.plugins.sglang.layers.dsv4_compression_metadata.reference import Dsv4CompressionMetadataReference

__all__ = ["CONTRACTS", "DSV4_COMPRESSION_METADATA"]

_EXACT = ToleranceBound(absolute_tolerance=0, relative_tolerance=0)
_INDEX = DTypeSpec(any_of=(DType.INT32, DType.INT64))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
# slots and positions stay far below 2**31 in sglang's pools
_SLOT_LIMIT = 1 << 30

DSV4_COMPRESSION_METADATA = OpContract(
    contract_id="dsv4_compression_metadata@1.0",
    description=(
        "sglang's DeepSeek-V4 compressed-cache metadata on HIP. Per row b and ratio r in (4, 128): out_loc_r = "
        "raw_out_loc // r when seq_lens % r == 0, else 0 (rows below n_write only); positions_r = positions with "
        "its low log2(r) bits cleared; seq_lens_raw_r = seq_lens // r; seq_lens_clamp1_r = max(seq_lens_raw_r, 1). "
        "c128_page_indices[b, j] = page_table[b, j // c128_page_size] * c128_page_size + j % c128_page_size for "
        "j < seq_lens_raw_128, else -1. Integer division truncates toward zero."
    ),
    op_type="dsv4_compression_metadata",
    axes={
        "n_rows": axis("Metadata rows, one per query token of the step.", 1, 65536, is_var=True),
        "n_write": axis("Rows that write a cache slot: the first n_write rows.", 1, 65536, is_var=True),
        "max_pages": axis("Page table columns.", 1, 1 << 20),
        "c128_width": axis("Ratio-128 page index columns: c128_page_size * max_pages.", 1, 64 << 20),
        "c128_page_size": derived_axis(
            "Ratio-128 slots per KV page: page_size // 128.", "c128_width // max_pages"
        ),
    },
    constraints=("n_write <= n_rows", "c128_width % max_pages == 0"),
    inputs={
        "seq_lens": TensorSpec(
            description="Each row's sequence length.",
            shape=("n_rows",),
            dtype=_INDEX,
            distribution=IntRangeDistribution(lo="1", hi="128 * c128_page_size * max_pages"),
        ),
        "positions": TensorSpec(
            description="Each row's position.",
            shape=("n_rows",),
            dtype=_INDEX,
            distribution=IntRangeDistribution(lo="0", hi=str(_SLOT_LIMIT)),
        ),
        "raw_out_loc": TensorSpec(
            description="The KV slot each writing row stores into.",
            shape=("n_write",),
            dtype=_INDEX,
            distribution=IntRangeDistribution(lo="0", hi=str(_SLOT_LIMIT)),
        ),
        "page_table": TensorSpec(
            description="Each row's KV pages.",
            shape=("n_rows", "max_pages"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi=str(_SLOT_LIMIT // 64)),
        ),
    },
    outputs={
        "c4_out_loc": TensorSpec(
            description="Ratio-4 slot of each writing row.",
            shape=("n_write",),
            dtype=_INT64,
            error_bounds=_EXACT,
        ),
        "c4_positions": TensorSpec(
            description="Ratio-4 positions.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c4_seq_lens_raw": TensorSpec(
            description="Ratio-4 lengths.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c4_seq_lens_clamp1": TensorSpec(
            description="Ratio-4 lengths, at least 1.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c128_out_loc": TensorSpec(
            description="Ratio-128 slot of each writing row.",
            shape=("n_write",),
            dtype=_INT64,
            error_bounds=_EXACT,
        ),
        "c128_positions": TensorSpec(
            description="Ratio-128 positions.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c128_seq_lens_raw": TensorSpec(
            description="Ratio-128 lengths.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c128_seq_lens_clamp1": TensorSpec(
            description="Ratio-128 lengths, at least 1.", shape=("n_rows",), dtype=_INT32, error_bounds=_EXACT
        ),
        "c128_page_indices": TensorSpec(
            description="Ratio-128 slot of each compressed position, -1 past the row's length.",
            shape=("n_rows", "c128_width"),
            dtype=_INT32,
            error_bounds=_EXACT,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=Dsv4CompressionMetadataReference(),
    baseline=Dsv4CompressionMetadataBaseline(),
)

CONTRACTS = (DSV4_COMPRESSION_METADATA,)
