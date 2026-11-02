# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``index_q_pack``'s gate GEMV: x . gate_w^T in fp32 (sglang's ``linear_bf16_fp32``
within summation-order noise), and the query's outputs the same with or without it."""

import pytest
import torch

pytestmark = pytest.mark.gpu

HEADS, QRANK, HIDDEN = 32, 1280, 5120


def _inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qr = torch.randn(rows, QRANK, generator=g, device="cuda").bfloat16()
    w = (torch.randn(HEADS * 128, QRANK, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    ws = torch.randint(120, 124, (HEADS * 4, QRANK // 32), generator=g, device="cuda").to(torch.uint8)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    wp = (torch.randn(HEADS, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    pos = torch.randint(0, 131072, (rows,), generator=g, device="cuda")
    table = torch.randn(1 << 18, 32, 2, generator=g, device="cuda")
    gate = (torch.randn(1024, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    return (qr, w.view(torch.uint8), ws, x, wp, pos, table, 0.125), gate


@pytest.mark.parametrize("rows", [1, 6, 8, 12])
@pytest.mark.parametrize("n", [1024, 512, 37])
def test_gate_gemv(rows, n):
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    args, gate = _inputs(rows, rows * 7 + n)
    gate = gate[:n].contiguous()
    plain = hipops.ops().index_q_pack(*args)
    got = hipops.ops().index_q_pack(*args, gate)
    assert len(plain) == 3 and len(got) == 4
    for a, b in zip(plain, got[:3]):
        assert torch.equal(a, b)
    want = torch.mm(args[3], gate.t(), out_dtype=torch.float32)
    assert got[3].shape == (rows, n) and got[3].dtype == torch.float32
    assert (got[3] - want).abs().max().item() <= 1e-4 * want.abs().max().item()


@pytest.mark.parametrize("rows", [9, 24, 48])
@pytest.mark.parametrize("gated", [False, True])
def test_wide_query_is_index_q_packs(rows, gated):
    """Past eight rows the Triton query writes index_q_pack's FP4 bytes and scales, its head
    weights within a bf16 step and its gate GEMV within summation-order noise."""
    from aklr.plugins.sglang.models.deepseek_v4 import index_q
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    args, gate = _inputs(rows, rows * 13 + gated)
    gate = gate if gated else None
    qr, w, ws, x, wp, pos, table, scale = args
    assert index_q.supported(qr, w, x, table)
    want = hipops.ops().index_q_pack(*args, gate)
    got = index_q.index_q(qr, w, ws, x, wp, pos, table, scale, gate)
    assert len(got) == len(want)
    assert [t.dtype for t in got] == [t.dtype for t in want]
    for a, b in zip(want[:2], got[:2]):
        assert (a == b).float().mean().item() >= 0.999
    assert torch.allclose(want[2].float(), got[2].float(), rtol=1e-2, atol=1e-6)
    if gated:
        assert (got[3] - want[3]).abs().max().item() <= 1e-4 * want[3].abs().max().item()
