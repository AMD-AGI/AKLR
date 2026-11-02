# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K2i on SGLang's page-8 index-K layout (SGLANG_HIP_DSV41_INDEX_K_PAGE8): the same keys,
written by SGLang's own writer in either layout, give the same logits bit for bit, on a
FULL layer's request page tables and on a REINDEX layer's candidate pages."""

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGES = 96  # 64-row pages of the plane


def _plane(keys, loc, page8):
    from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache_hip import (
        store_fp4_index_k_cache_split,
    )

    payload = torch.zeros(PAGES, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(PAGES, 1, 4, 64, dtype=torch.uint8, device="cuda")
    store_fp4_index_k_cache_split(keys, payload, scale, loc, page_size=64, rne=True, page8=page8)
    return payload, scale, page8


def _query(rows, g):
    q = torch.randint(0, 256, (rows, 32, 64), generator=g, device="cuda").to(torch.uint8)
    qs = torch.randint(118, 126, (rows, 1, 4, 16, 4), generator=g, device="cuda")
    w = (torch.randn(rows, 32, generator=g, device="cuda") * 0.1).bfloat16()
    return q, qs.to(torch.uint8), w


def _setup(seed):
    pytest.importorskip("sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache_hip")
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.indexer.runner import (
        K2iRunner,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randn(PAGES * 64, 128, generator=g, device="cuda").bfloat16()
    loc = torch.arange(PAGES * 64, device="cuda")
    planes = [_plane(keys, loc, page8) for page8 in (False, True)]
    return K2iRunner(torch.device("cuda"), bound_max=PAGES * 64), planes, g


@pytest.mark.parametrize("reqs,per_req", [(1, 6), (2, 6), (8, 6), (3, 1)])
def test_full_layer_logits_are_the_page64_ones(reqs, per_req):
    runner, planes, g = _setup(reqs * 10 + per_req)
    rows, cols = reqs * per_req, 40
    # each request's own pages, in no order; its rows see one more column each (verify)
    tables = torch.stack([torch.randperm(PAGES, generator=g, device="cuda")[:cols] for _ in range(reqs)])
    page_table = tables.repeat_interleave(per_req, 0).to(torch.int32).contiguous()
    lens = torch.randint(600, cols * 64 - per_req, (reqs,), generator=g, device="cuda")
    bounds = (lens.repeat_interleave(per_req) + torch.arange(rows, device="cuda") % per_req).to(torch.int32)
    query = _query(rows, g)
    got = []
    for plane in planes:
        runner.walk_plan(ratio=1, bounds=bounds, rows=rows, per_req=per_req)
        out = runner.score(
            rows=rows, per_req=per_req, ratio=1, query=query, plane=plane,
            page_table=page_table, bounds=bounds,
        )  # fmt: skip
        got.append(out.clone())
    torch.cuda.synchronize()
    for r in range(rows):
        b = int(bounds[r])
        assert torch.equal(got[0][r, :b], got[1][r, :b]), r
        assert got[1][r, :b].abs().max().item() > 0


@pytest.mark.parametrize("rows", [1, 6, 12])
def test_reindex_layer_logits_are_the_page64_ones(rows):
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.indexer import (
        index_plan as ip,
    )
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.indexer.runner import (
        COMPACT,
    )

    runner, planes, g = _setup(rows)
    # a row's candidates: virtual 8-row pages of the plane (ishift 0)
    table = torch.randint(0, PAGES * 8, (rows, ip.TOPK_BLOCKS), generator=g, device="cuda")
    table = table.to(torch.int32).contiguous()
    lens = torch.randint(ip.TOPK + 1, 3000, (rows,), generator=g, device="cuda").to(torch.int32)
    query = _query(rows, g)
    got = []
    for plane in planes:
        runner._score(
            rows=rows, query=query, plane=plane, plan=runner.plans[1], itab=table,
            itab_stride=ip.TOPK_BLOCKS, itab_len=ip.TOPK_BLOCKS, ibatch=runner.owners[0],
            ishift=0, ibound=lens, lstride=COMPACT, reindex=1,
        )  # fmt: skip
        got.append(runner.ilog[: rows * COMPACT].view(rows, COMPACT).clone())
    torch.cuda.synchronize()
    for r in range(rows):
        b = int(lens[r])
        assert torch.equal(got[0][r, :b], got[1][r, :b]), r
        assert got[1][r, :b].abs().max().item() > 0
