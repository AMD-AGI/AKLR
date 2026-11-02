# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``deepseek_v4.prologue``'s kernels write the integers of sglang's torch code (as in
``deepseek_v4_backend_hip_radix``: ``expand_extend_with_same_length``,
``get_swa_page_indices`` with ``DeepSeekV4TokenToKVPool``'s translation)."""

import pytest
import torch

pytestmark = pytest.mark.gpu

WINDOW = 128


def _sglang_expand(bs, qo_len, seq_lens, req_pool_indices):
    i32 = {"dtype": torch.int32, "device": seq_lens.device}
    seq_lens_casual = seq_lens[:, None] + torch.arange(-qo_len + 1, 1, **i32)
    idx = torch.arange(bs, **i32).repeat_interleave(qo_len)
    return seq_lens_casual.flatten(), req_pool_indices[idx]


def _sglang_swa(seq_lens_casual, req_pool_indices_repeated, req_to_token, mapping):
    pos = seq_lens_casual - 1
    offsets = pos.unsqueeze(1) - torch.arange(WINDOW, dtype=torch.int32, device=pos.device).unsqueeze(0)
    invalid = offsets < 0
    offsets.masked_fill_(invalid, 0)
    raw = req_to_token[req_pool_indices_repeated[:, None], offsets]
    raw.masked_fill_(invalid, -1)
    return mapping[raw].to(torch.int32)


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("bs,qo_len", [(1, 6), (2, 6), (8, 5), (48, 1)])
def test_expand_is_sglangs(seq_dtype, bs, qo_len):
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    g = torch.Generator(device="cuda").manual_seed(bs * 7 + qo_len)
    seq = torch.randint(1, 1 << 20, (bs,), device="cuda", generator=g).to(seq_dtype)
    req = torch.randperm(64, device="cuda", generator=g)[:bs].to(torch.int64)
    got = prologue.expand(bs, qo_len, seq, req)
    want = _sglang_expand(bs, qo_len, seq, req)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_expand_reads_strided_inputs():
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    seq = torch.arange(10, 30, device="cuda", dtype=torch.int64)[::2]
    req = torch.arange(100, 120, device="cuda", dtype=torch.int32)[::2]
    for a, b in zip(prologue.expand(10, 6, seq, req), _sglang_expand(10, 6, seq, req)):
        assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 6, 12, 48])
def test_swa_page_indices_are_sglangs(map_dtype, rows):
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    g = torch.Generator(device="cuda").manual_seed(rows)
    slots, ctx, tokens = 16, 4096, 1 << 16
    req_to_token = torch.randint(0, tokens, (slots, ctx), device="cuda", generator=g, dtype=torch.int32)
    mapping = torch.randint(0, 1 << 30, (tokens + 1,), device="cuda", generator=g).to(map_dtype)
    # rows near a request's start read slots before it: -1, the mapping's last entry
    seq = torch.randint(1, ctx, (rows,), device="cuda", generator=g, dtype=torch.int32)
    seq[: rows // 2] = torch.randint(1, WINDOW, (rows // 2,), device="cuda", generator=g, dtype=torch.int32)
    req = torch.randint(0, slots, (rows,), device="cuda", generator=g, dtype=torch.int64)
    got = prologue.swa_page_indices(
        seq_lens_casual=seq,
        req_pool_indices_repeated=req,
        req_to_token=req_to_token,
        mapping=mapping,
        window=WINDOW,
    )
    want = _sglang_swa(seq, req, req_to_token, mapping)
    assert prologue.swa_disagrees(want, got) is None
