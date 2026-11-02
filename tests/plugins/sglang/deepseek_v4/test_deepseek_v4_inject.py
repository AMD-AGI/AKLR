# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``swa_store``: every stage's V4 / V4.1 SWA cache bytes in one launch are the bytes of
sglang's ``fused_k_norm_rope_flashmla`` a stage at a time, the stages' rows read off one
strided GEMM output; rows without a slot are left alone."""

import pytest
import torch

pytestmark = pytest.mark.gpu

DIM, ROPE, STAGES = 512, 64, 3
# (row bytes, scale bytes, page alignment) of a token
LAYOUTS = {"v4": (576, 8, 576), "v41": (512, 16, 512)}


def _sglang_store():
    try:
        from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    except ImportError as e:
        pytest.skip(f"sglang's DeepSeek-V4 kernels: {e}")
    return fused_k_norm_rope_flashmla, KVLayout


def _page_bytes(layout, page_size):
    row, scale, align = LAYOUTS[layout]
    return -(-page_size * (row + scale) // align) * align


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("rows", [1, 6, 48])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
def test_swa_store_matches_sglang(layout, page_size, rows, pos_dtype):
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    store, kv_layout = _sglang_store()
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + page_size)
    kv_all = (torch.randn(rows, STAGES * DIM, generator=g, device="cuda") * 3.0).bfloat16()
    kv_all[0, 5] = 40.0  # an outlier in one tile
    kvs = list(kv_all.split(DIM, dim=-1))
    weights = [(1.0 + 0.2 * torch.randn(DIM, generator=g, device="cuda")).bfloat16() for _ in range(STAGES)]
    positions = 4096
    angles = torch.rand(positions, ROPE // 2, generator=g, device="cuda") * 6.28
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    pages = 8
    pos = torch.randint(0, positions, (rows,), generator=g, device="cuda").to(pos_dtype)
    loc = torch.randperm(pages * page_size, generator=g, device="cuda")[:rows]
    loc = loc.to(torch.int32)
    if rows > 1:
        loc[1] = -1
    want = [
        torch.randint(0, 256, (pages, _page_bytes(layout, page_size)), generator=g, device="cuda").to(
            torch.uint8
        )
        for _ in range(STAGES)
    ]
    got = [c.clone() for c in want]
    for s in range(STAGES):
        store(
            kv=kvs[s].contiguous(),
            kv_weight=weights[s],
            eps=1e-6,
            freqs_cis=freqs_cis,
            positions=pos,
            out_loc=loc,
            kvcache=want[s],
            page_size=page_size,
            layout=kv_layout.parse(layout),
        )
    hipops.ops().swa_store(
        kvs[0],
        DIM,
        weights,
        [freqs] * STAGES,
        pos,
        loc,
        got,
        1e-6,
        page_size,
        layout == "v41",
    )
    torch.cuda.synchronize()
    for s in range(STAGES):
        assert torch.equal(got[s], want[s]), f"stage {s}"
