# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests of the committed FlyDSL kernels at the sizes their any_size axes route to them."""

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import BuilderRegistry
from aklr.plugins.aiter.mla.reference import fp8_paged_mqa_logits_reference

_MQA_SCHEMA = "fp8_paged_mqa_logits_page_size64_n_heads32"


def _flydsl_schema(name: str):
    schemas, _ = ContractRegistry.default().read_schemas("aklr.plugins.aiter.mla")
    (schema,) = [
        s for _, s in schemas if s.name == name and s.kernel is not None and s.kernel.language == "flydsl"
    ]
    return schema


@pytest.mark.gfx950
@pytest.mark.parametrize(
    ("batch", "n_pages", "max_blocks", "weight_scale"),
    [
        (72, 65536, 32, 1.0),
        (24, 254200, 32, 1.0),
        (192, 50264, 16448, 1.0),
        (96, 2**21 + 1, 64, 1.0),
        (24, 1024, 16, 2.0**120),
    ],
    ids=[
        "bands_past_1024_pages",
        "pool_past_2GiB",
        "table_at_contract_bound",
        "pool_past_2M_pages",
        "huge_weights",
    ],
)
def test_fp8_paged_mqa_logits_serves_any_pool_and_table(
    builder_registry: BuilderRegistry, batch: int, n_pages: int, max_blocks: int, weight_scale: float
) -> None:
    """A pool or table of any contract size, and any finite weights, give the reference's logits: each
    request's partial last page ends in -inf, and nothing in its context is lost to overflow."""
    schema = _flydsl_schema(_MQA_SCHEMA)
    kernel = builder_registry.build(schema)
    gen = torch.Generator(device="cuda").manual_seed(0)
    table = torch.randint(0, n_pages, (batch, max_blocks), dtype=torch.int32, device="cuda", generator=gen)
    # only the pages the tables hold are written, and the reference scores a pool of just those
    used, compact = torch.unique(table.long(), return_inverse=True)
    pool = torch.empty(n_pages, 64, 1, 132, dtype=torch.uint8, device="cuda")
    pool[used] = torch.randint(
        0, 64, (used.numel(), 64, 1, 132), dtype=torch.uint8, device="cuda", generator=gen
    )
    q = torch.randn(batch, 1, 32, 128, device="cuda", generator=gen).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, 32, device="cuda", generator=gen) * weight_scale
    context = torch.randint(1, max_blocks * 64 + 1, (batch,), dtype=torch.int32, device="cuda", generator=gen)
    logits = torch.full((batch, max_blocks * 64), torch.nan, device="cuda")
    kernel(
        q=q, kv_cache=pool, weights=weights, context_lens=context, block_tables=table,
        logits=logits, page_size=64, preshuffled=True,
    )  # fmt: skip
    axes = {"batch": batch, "n_pages": used.numel(), "max_blocks": max_blocks, "max_seq_len": max_blocks * 64}
    args = {
        "q": q, "kv_cache": pool[used], "weights": weights, "context_lens": context,
        "block_tables": compact.int(),
    }  # fmt: skip
    expected = fp8_paged_mqa_logits_reference(schema.point(axes), args, lambda t, _dtype: t.double())
    tokens = torch.arange(max_blocks * 64, device="cuda")
    live = tokens < context[:, None]
    tail = ~live & (tokens < (context[:, None] + 63) // 64 * 64)
    assert torch.isfinite(logits[live]).all()
    assert (logits[tail] == -torch.inf).all()
    error = (logits[live].double() - expected[live]).square().sum() / expected[live].square().sum()
    assert error < 1e-3, error
