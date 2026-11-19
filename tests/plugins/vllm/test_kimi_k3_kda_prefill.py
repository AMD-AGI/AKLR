# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route of Kimi-K3's KDA prefill: how vLLM's chunk_kda_prefill call maps onto the op, and
which calls stay with vLLM."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.kda_prefill import api as kda_api
from aklr.plugins.vllm.models.kimi_k3.kda_prefill.reference import kimi_k3_kda_prefill_reference

HEADS, DIM, SLOTS, CONV = 2, 128, 3, 96
SEQLENS = (5, 3)

# vLLM 0.34.0's and 0.32.1rc1.dev276's models/kimi_k3/amd/ops/kda_prefill.py
VLLM_PARAMETERS = (
    "q k v raw_g raw_beta A_log g_bias scale initial_state output_final_state lower_bound "
    "use_qk_l2norm_in_kernel cu_seqlens chunk_indices chunk_offsets backend out checkpoint_state "
    "checkpoint_offsets checkpoint_state_indices state_cache state_indices has_initial_state"
).split()


def test_the_route_takes_vllms_parameters():
    assert list(inspect.signature(kda_api._chunk_kda_prefill).parameters) == VLLM_PARAMETERS


def vllm_call(direct_out):
    """A fused-backend call as kda.py makes it: [1, T, H, D] views, beta a column slice of the
    fused projection, the recurrent state in pages shared with the convolution state."""
    gen = torch.Generator().manual_seed(0)
    tokens = sum(SEQLENS)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    projection = randn(tokens, HEADS * DIM + HEADS)
    pages = torch.randn(SLOTS, HEADS * DIM * DIM + CONV, generator=gen)
    out = torch.empty(1, tokens + 2, HEADS, DIM, dtype=torch.bfloat16)
    return dict(
        q=randn(1, tokens, HEADS, DIM),
        k=randn(1, tokens, HEADS, DIM),
        v=randn(1, tokens, HEADS, DIM),
        raw_g=randn(1, tokens, HEADS, DIM, scale=DIM**-0.5),
        raw_beta=projection[:, HEADS * DIM :].unsqueeze(0),
        A_log=torch.rand(HEADS, generator=gen).add(1).log(),
        g_bias=torch.randn(HEADS * DIM, generator=gen),
        lower_bound=-5.0,
        state_cache=pages[:, : HEADS * DIM * DIM].view(SLOTS, HEADS, DIM, DIM),
        state_indices=torch.tensor([2, 0], dtype=torch.int32),
        has_initial_state=torch.tensor([True, False]),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, SEQLENS[0], tokens], dtype=torch.int32),
        backend="fused",
        out=out[:, 2:] if direct_out else None,
    ), pages


def expected(call):
    """vLLM's semantics, written apart from the contract's reference."""
    q, k, v = (call[n][0].float() for n in "qkv")
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * DIM**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    g_bias = call["g_bias"].view(HEADS, DIM)
    gate = call["lower_bound"] * torch.sigmoid(
        call["A_log"].exp()[:, None] * (call["raw_g"][0].float() + g_bias)
    )
    beta = torch.sigmoid(call["raw_beta"][0].float())
    states = call["state_cache"].clone()
    out = torch.empty_like(v)
    bounds = call["cu_seqlens"].tolist()
    for s, (slot, resume) in enumerate(
        zip(call["state_indices"].tolist(), call["has_initial_state"].tolist())
    ):
        S = states[slot].clone() if resume else torch.zeros(HEADS, DIM, DIM)
        for t in range(bounds[s], bounds[s + 1]):
            S = S * gate[t].exp()[:, None, :]
            S = (
                S
                + beta[t][:, None, None]
                * (v[t] - (S @ k[t][..., None])[..., 0])[..., None]
                * k[t][:, None, :]
            )
            out[t] = (S @ q[t][..., None])[..., 0]
        states[slot] = S
    return out, states


class _Op:
    """An op as the route sees it: dispatch's question first (``reason``: None when it serves), then the call."""

    def __init__(self, fn, reason=None):
        self.fn, self.reason = fn, reason

    def fallback_reason(self, *args, **kwargs):
        return self.reason

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@pytest.fixture(autouse=True)
def _no_remembered_refusals(monkeypatch):
    """Each test asks the route afresh: the route remembers the table's refusals and services per shape, and keeps
    staging rows and plans."""
    from aklr.plugins.vllm.models.kimi_k3.kda_prefill import api as route_api

    monkeypatch.setattr(route_api, "_DECLINED", {})
    monkeypatch.setattr(route_api, "_SERVED", set())
    monkeypatch.setattr(route_api, "_STAGING", {})


@pytest.fixture
def op_runs_reference(monkeypatch):
    """The op as its reference computes it, writing state_cache and out as a kernel does."""

    def op(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        state_storage="row_major",
        scale=DIM**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        assert state_storage == "strided_rows" and raw_beta.is_contiguous()
        assert state_cache.shape == (SLOTS, HEADS * DIM, DIM) and g_bias.shape == (HEADS, DIM)
        args = dict(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            a_log=a_log,
            g_bias=g_bias,
            state_cache=state_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            lower_bound=lower_bound,
        )
        point = SimpleNamespace(axes={"n_heads": HEADS, "head_dim": DIM}, dtypes=dict.fromkeys(args, None))
        o, states = kimi_k3_kda_prefill_reference(point, args, lambda x, _: x.float())
        state_cache.copy_(states)
        if out is None:
            return o.to(q.dtype)
        return out.copy_(o)

    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill", _Op(op))


@pytest.mark.parametrize("direct_out", [True, False], ids=["into_out", "own_out"])
def test_a_fused_call_runs_the_op_on_vllms_paged_state(op_runs_reference, direct_out):
    call, pages = vllm_call(direct_out)
    conv_before, untouched_before = pages[:, HEADS * DIM * DIM :].clone(), pages[1].clone()
    want_out, want_states = expected(call)
    out, final_state = kda_api._chunk_kda_prefill(**call)
    assert final_state is None and out.shape == (1, sum(SEQLENS), HEADS, DIM)
    if direct_out:
        assert out.data_ptr() == call["out"].data_ptr()
    torch.testing.assert_close(out[0].float(), want_out, atol=2e-3, rtol=2e-2)
    for slot in call["state_indices"].tolist():
        torch.testing.assert_close(call["state_cache"][slot], want_states[slot], atol=1e-4, rtol=1e-4)
    assert torch.equal(pages[1], untouched_before)  # a slot no sequence names
    assert torch.equal(pages[:, HEADS * DIM * DIM :], conv_before)  # the convolution state


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "triton"},
        {"backend": "gluon"},
        {"use_qk_l2norm_in_kernel": False},
        {"state_cache": None, "initial_state": torch.zeros(2, HEADS, DIM, DIM)},
        {"output_final_state": True},
        {"g_bias": None},
    ],
    ids=[
        "triton",
        "gluon",
        "no_l2norm",
        "no_cache",
        "final_state",
        "no_bias",
    ],
)
def test_other_calls_stay_with_vllm(monkeypatch, change):
    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = vllm_call(direct_out=True)
    call.update(change)
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.OUT_OF_CONTRACT


def test_a_state_not_dense_per_slot_stays_with_vllm(monkeypatch):
    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = vllm_call(direct_out=True)
    call["state_cache"] = torch.zeros(SLOTS, DIM, HEADS, DIM).transpose(1, 2)
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.OUT_OF_CONTRACT


def test_a_declined_op_call_goes_back_to_vllm(monkeypatch):
    monkeypatch.setattr(
        kda_api,
        "kimi_k3_kda_prefill",
        _Op(lambda *a, **kw: pytest.fail("the op ran"), FallbackReason.NO_WORKLOAD),
    )
    call, _ = vllm_call(direct_out=True)
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.NO_WORKLOAD


def test_a_shape_served_last_time_skips_the_question_until_dispatch_declines_it(monkeypatch):
    asked, answers = [], [None, None, FallbackReason.NO_WORKLOAD, None]

    class Op(_Op):
        def fallback_reason(self, *args, **kwargs):
            asked.append(True)
            return None

        def __call__(self, *args, **kwargs):  # dispatch: asks again, serves or declines
            reason = answers.pop(0)
            assert args[4].is_contiguous()  # beta, copied to be served
            return reason if reason is not None else kwargs["out"]

    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill", Op(None))
    call, _ = vllm_call(direct_out=True)
    assert kda_api._chunk_kda_prefill(**call)[0] is call["out"] and len(asked) == 1
    assert kda_api._chunk_kda_prefill(**call)[0] is call["out"] and len(asked) == 1  # served: not asked
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.NO_WORKLOAD and len(asked) == 1
    # declined since: asked again
    assert kda_api._chunk_kda_prefill(**call)[0] is call["out"] and len(asked) == 2


# vLLM's mid-prefill snapshot: kda.py passes the paged cache itself as checkpoint_state, each sequence's offset into its
# query (a multiple of 64, 0 for none) and the cache row to write it to (NULL_BLOCK_ID for none)
CK_SLOTS, NULL = 5, 0


def checkpoint_call(offsets, rows):
    call, _ = vllm_call(direct_out=True)
    gen = torch.Generator().manual_seed(1)
    pages = torch.randn(CK_SLOTS, HEADS * DIM * DIM + CONV, generator=gen)
    cache = pages[:, : HEADS * DIM * DIM].view(CK_SLOTS, HEADS, DIM, DIM)
    call.update(
        state_cache=cache,
        state_indices=torch.tensor([2, 3], dtype=torch.int32),
        checkpoint_state=cache,
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32),
        checkpoint_state_indices=torch.tensor(rows, dtype=torch.int32),
    )
    return call, pages


def expected_with_snapshots(call):
    """vLLM's semantics: as ``expected``, and each taken snapshot in its cache row."""
    q, k, v = (call[n][0].float() for n in "qkv")
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * DIM**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = call["lower_bound"] * torch.sigmoid(
        call["A_log"].exp()[:, None] * (call["raw_g"][0].float() + call["g_bias"].view(HEADS, DIM))
    )
    beta = torch.sigmoid(call["raw_beta"][0].float())
    states = call["state_cache"].clone()
    out = torch.empty_like(v)
    bounds = call["cu_seqlens"].tolist()
    for s, slot in enumerate(call["state_indices"].tolist()):
        S = states[slot].clone() if call["has_initial_state"][s] else torch.zeros(HEADS, DIM, DIM)
        at = bounds[s] + call["checkpoint_offsets"][s].item()
        row = call["checkpoint_state_indices"][s].item()
        for t in range(bounds[s], bounds[s + 1]):
            S = S * gate[t].exp()[:, None, :]
            S = (
                S
                + beta[t][:, None, None]
                * (v[t] - (S @ k[t][..., None])[..., 0])[..., None]
                * k[t][:, None, :]
            )
            out[t] = (S @ q[t][..., None])[..., 0]
            # vLLM's kernel skips an offset that is not a whole number of 64-token chunks
            if at > bounds[s] and (at - bounds[s]) % 64 == 0 and t + 1 == at and row != NULL:
                states[row] = S
        states[slot] = S
    return out, states


@pytest.fixture
def checkpoint_op_runs_reference(monkeypatch):
    """The checkpoint op as its reference computes it."""
    from aklr.plugins.vllm.models.kimi_k3.kda_prefill.reference import (
        kimi_k3_kda_prefill_checkpoint_reference,
    )

    def op(q, k, v, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, has_initial_state, cu_seqlens,
           checkpoint_chunks, checkpoint_state, state_storage="row_major", scale=DIM**-0.5, lower_bound=-5.0,
           out_dtype=None, out=None):  # fmt: skip
        assert state_storage == "strided_rows" and checkpoint_state.is_contiguous()
        # the staging rows of sequences without a snapshot stay as they were and land in the null row
        args = dict(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, a_log=a_log, g_bias=g_bias,
                    state_cache=state_cache, state_indices=state_indices, has_initial_state=has_initial_state,
                    cu_seqlens=cu_seqlens, checkpoint_chunks=checkpoint_chunks, checkpoint_state=checkpoint_state,
                    scale=scale, lower_bound=lower_bound)  # fmt: skip
        point = SimpleNamespace(axes={"n_heads": HEADS, "head_dim": DIM}, dtypes=dict.fromkeys(args, None))
        o, states, snaps = kimi_k3_kda_prefill_checkpoint_reference(point, args, lambda x, _: x.float())
        state_cache.copy_(states)
        checkpoint_state.copy_(snaps)
        return out.copy_(o)

    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill_checkpoint", _Op(op))
    monkeypatch.setattr(
        kda_api, "kimi_k3_kda_prefill", lambda *a, **kw: pytest.fail("the op without snapshots ran")
    )


@pytest.mark.parametrize(
    ("offsets", "rows"),
    [
        ([64, 0], [1, NULL]),
        ([0, 64], [NULL, 4]),
        ([64, 64], [1, 4]),
        ([64, 0], [NULL, NULL]),
        ([3, 0], [1, NULL]),
    ],
    ids=["first", "second", "both", "no_row", "not_a_chunk"],
)
def test_a_call_with_snapshots_takes_them_into_the_cache_rows(checkpoint_op_runs_reference, offsets, rows):
    # the two sequences: 5 and 3 tokens in vllm_call; long enough here for a 64-token chunk
    global SEQLENS
    saved, SEQLENS = SEQLENS, (70, 66)
    try:
        call, pages = checkpoint_call(offsets, rows)
    finally:
        SEQLENS = saved
    conv_before = pages[:, HEADS * DIM * DIM :].clone()
    want_out, want_states = expected_with_snapshots(call)
    out, final_state = kda_api._chunk_kda_prefill(**call)
    assert final_state is None
    torch.testing.assert_close(out[0].float(), want_out, atol=2e-3, rtol=2e-2)
    written = {2, 3} | {r for o, r in zip(offsets, rows) if r != NULL and o % 64 == 0 and o > 0}
    for slot in written:
        torch.testing.assert_close(call["state_cache"][slot], want_states[slot], atol=1e-4, rtol=1e-4)
    untouched = set(range(1, CK_SLOTS)) - written
    for slot in untouched:
        assert torch.equal(call["state_cache"][slot], want_states[slot])
    assert torch.equal(pages[:, HEADS * DIM * DIM :], conv_before)


def test_a_partial_checkpoint_stays_with_vllm(monkeypatch):
    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill_checkpoint", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = checkpoint_call([64, 0], [1, NULL])
    call["checkpoint_state_indices"] = None
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.OUT_OF_CONTRACT
