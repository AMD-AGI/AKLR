# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``dspark_accept_xchg`` on one rank: the tokens are ``torch.argmax`` over the valid columns
(lowest index on ties), and the accept buffers are sglang's greedy DSpark accept of the
candidates against them, eager and replayed from a CUDA graph."""

import pytest
import torch

pytestmark = pytest.mark.gpu

VOCAB_PART = 64640
T = 6  # the anchor and 5 drafts


def _want(logits, valid, cand, prefix):
    bs, n = cand.shape
    tok = logits[:, :valid].float().argmax(dim=-1).view(bs, n)
    correct = torch.zeros(bs, dtype=torch.int64, device=cand.device)
    for s in range(bs):
        while correct[s] < n - 1 and cand[s, correct[s] + 1] == tok[s, correct[s]]:
            correct[s] += 1
    rows = torch.arange(bs, device=cand.device)
    bonus = tok[rows, correct]
    out = torch.zeros(bs, n, dtype=torch.int64, device=cand.device)
    out[:, : n - 1] = cand[:, 1:]
    out[rows, correct] = bonus
    commit = (correct + 1).to(torch.int32)
    return tok.view(-1), correct, bonus, commit, prefix.long() + correct + 1, out


def _case(bs, dtype, valid, seed, prefix_dtype):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(bs * T, VOCAB_PART, generator=g, device="cuda").to(dtype)
    if valid < VOCAB_PART:  # padding columns that would win if read
        logits[:, valid:] = 100.0
    # a tie: the lower index is the token
    logits[0, 7] = logits[0, 9] = 50.0
    tok = logits[:, :valid].float().argmax(dim=-1).view(bs, T)
    cand = torch.randint(0, valid, (bs, T), generator=g, device="cuda")
    for s in range(bs):
        matched = (s * 2 + seed) % T  # 0 .. T - 1 leading matches
        cand[s, 1 : matched + 1] = tok[s, :matched]
        if matched < T - 1:
            cand[s, matched + 1] = (tok[s, matched] + 1) % valid
    prefix = torch.randint(1000, 200000, (bs,), generator=g, device="cuda").to(prefix_dtype)
    return logits, cand, prefix


def _bufs(bs):
    i64 = {"dtype": torch.int64, "device": "cuda"}
    i32 = {"dtype": torch.int32, "device": "cuda"}
    return [
        torch.full((bs,), -1, **i64),
        torch.full((bs,), -1, **i64),
        torch.full((bs,), -1, **i32),
        torch.full((bs,), -1, **i32),
        torch.full((bs,), -1, **i64),
        torch.full((bs, T), -1, **i64),
    ]


def _check(got_tok, bufs, want):
    tok, correct, bonus, commit, new_lens, out = want
    got_correct, got_bonus, got_cap, got_commit, got_new, got_out = bufs
    assert torch.equal(got_tok, tok)
    assert torch.equal(got_correct, correct)
    assert torch.equal(got_bonus, bonus)
    assert torch.equal(got_cap, torch.zeros_like(got_cap))
    assert torch.equal(got_commit, commit)
    assert torch.equal(got_new, new_lens)
    assert torch.equal(got_out, out)


@pytest.mark.parametrize("bs", [1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("valid", [VOCAB_PART, VOCAB_PART - 77])
@pytest.mark.parametrize("prefix_dtype", [torch.int64, torch.int32])
def test_accept_matches_sglang(bs, dtype, valid, prefix_dtype):
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    ops, x = hipops.ops(), hipops.Exchange(None, 0, 1, torch.device("cuda"))
    for seed in range(3):  # the call counter advances
        logits, cand, prefix = _case(bs, dtype, valid, seed, prefix_dtype)
        bufs = _bufs(bs)
        tok = ops.dspark_accept_xchg(logits, valid, 0, x.peers, x.argmax_calls, cand, prefix, *bufs)
        _check(tok, bufs, _want(logits, valid, cand, prefix))


@pytest.mark.parametrize("gate", [0, 1])
def test_accept_gates_the_commit(gate):
    """The static epilogue's folded commit lengths: min(commit, verify_lens) * gate, as
    sglang's torch ops."""
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    bs, valid = 8, VOCAB_PART
    ops, x = hipops.ops(), hipops.Exchange(None, 0, 1, torch.device("cuda"))
    logits, cand, prefix = _case(bs, torch.bfloat16, valid, 1, torch.int64)
    bufs = _bufs(bs)
    verify_lens = torch.tensor([T, 1, 3, T, 2, T, 0, 4], device="cuda")
    gate_buf = torch.full((1,), gate, dtype=torch.int32, device="cuda")
    gated = torch.full((bs,), -1, dtype=torch.int32, device="cuda")
    tok = ops.dspark_accept_xchg(
        logits, valid, 0, x.peers, x.argmax_calls, cand, prefix, *bufs,
        verify_lens, gate_buf, gated,
    )  # fmt: skip
    want = _want(logits, valid, cand, prefix)
    _check(tok, bufs, want)
    assert torch.equal(gated, torch.minimum(want[3], verify_lens.to(torch.int32)) * gate_buf)


def test_accept_replays():
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    bs, valid = 4, VOCAB_PART
    ops, x = hipops.ops(), hipops.Exchange(None, 0, 1, torch.device("cuda"))
    logits, cand, prefix = _case(bs, torch.bfloat16, valid, 0, torch.int64)
    bufs = _bufs(bs)
    ops.dspark_accept_xchg(logits, valid, 0, x.peers, x.argmax_calls, cand, prefix, *bufs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tok = ops.dspark_accept_xchg(logits, valid, 0, x.peers, x.argmax_calls, cand, prefix, *bufs)
    for seed in range(1, 4):
        new_logits, new_cand, new_prefix = _case(bs, torch.bfloat16, valid, seed, torch.int64)
        logits.copy_(new_logits), cand.copy_(new_cand), prefix.copy_(new_prefix)
        graph.replay()
        torch.cuda.synchronize()
        _check(tok, bufs, _want(logits, valid, cand, prefix))
