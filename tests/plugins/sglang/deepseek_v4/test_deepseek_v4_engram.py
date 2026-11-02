# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``deepseek_v4.engram.gate_keeping`` writes the bits of sglang's fused Engram gate and
its image-row ``torch.where`` (``DeepseekV4Model.forward``)."""

import pytest
import torch

pytestmark = pytest.mark.gpu

IMAGE = 129264


def _inputs(rows, hc, dim, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hc, dim, generator=g, device="cuda").to(dtype)
    kv = (torch.randn(rows, (hc + 1) * dim, generator=g, device="cuda") * 2).to(dtype)
    qw = (1 + 0.1 * torch.randn(hc, dim, generator=g, device="cuda")).to(dtype)
    kw = (1 + 0.1 * torch.randn(hc, dim, generator=g, device="cuda")).to(dtype)
    ids = torch.randint(0, 1000, (rows,), generator=g, device="cuda")
    ids[1::3] = IMAGE
    return x, kv, qw, kw, ids


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,hc,dim",
    [(1, 4, 5120), (6, 4, 5120), (12, 4, 5120), (6, 4, 4096), (5, 2, 1000)],
)
def test_gate_keeping_is_sglangs(dtype, rows, hc, dim):
    gate = pytest.importorskip("sglang.kernels.ops.embeddings.engram_gate")
    from aklr.plugins.sglang.models.deepseek_v4 import engram

    x, kv, qw, kw, ids = _inputs(rows, hc, dim, dtype, rows * 31 + dim)
    assert engram.applies(x, kv, qw, kw, ids)
    want = gate.fused_engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
    want = torch.where((ids == IMAGE)[:, None, None], x, want)
    got = engram.gate_keeping(x, kv, qw, kw, 1e-6, 1e-6, ids, IMAGE)
    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.view(bits), want.view(bits))
    assert torch.equal(got[1::3], x[1::3])


def test_applies_wants_one_id_a_row():
    from aklr.plugins.sglang.models.deepseek_v4 import engram

    x, kv, qw, kw, ids = _inputs(6, 4, 256, torch.bfloat16, 0)
    assert not engram.applies(x, kv, qw, kw, ids[:5])
    assert not engram.applies(x, kv[:, :-1].contiguous(), qw, kw, ids)
    assert not engram.applies(x, kv.float(), qw, kw, ids)
