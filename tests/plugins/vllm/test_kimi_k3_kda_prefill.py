# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route of Kimi-K3's KDA prefill: how vLLM's chunk_kda_prefill call maps onto the op, and
which calls stay with vLLM."""

import inspect
import json
import sys
import types

import pytest
import torch

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.kda_prefill import api as kda_api
from aklr.plugins.vllm.models.kimi_k3.kda_prefill.reference import KimiK3KdaPrefillReference
from tests.op_case import OpCase

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
        result = KimiK3KdaPrefillReference().reference(
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
        state_cache.copy_(result["state_cache"])
        if out is None:
            return result["out"]
        return out.copy_(result["out"])

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


def test_while_exporting_every_call_is_asked(monkeypatch):
    """With AKLR_EXPORT=1, read through RuntimeEnv when the route first needs it, dispatch counts every call: the
    route remembers no refusal and asks again."""
    asked = []

    class Op(_Op):
        def fallback_reason(self, *args, **kwargs):
            asked.append(True)
            return FallbackReason.NO_WORKLOAD

    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill", Op(None))
    monkeypatch.setenv("AKLR_EXPORT", "1")
    kda_api._remembers.cache_clear()
    try:
        call, _ = vllm_call(direct_out=True)
        for _ in range(2):
            assert kda_api._chunk_kda_prefill(**call) is FallbackReason.NO_WORKLOAD
    finally:
        kda_api._remembers.cache_clear()
    assert len(asked) == 2


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
    from aklr.plugins.vllm.models.kimi_k3.kda_prefill.reference import KimiK3KdaPrefillCheckpointReference

    def op(q, k, v, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, has_initial_state, cu_seqlens,
           checkpoint_chunks, checkpoint_state, state_storage="row_major", scale=DIM**-0.5, lower_bound=-5.0,
           out_dtype=None, out=None):  # fmt: skip
        assert state_storage == "strided_rows" and checkpoint_state.is_contiguous()
        # the staging rows of sequences without a snapshot stay as they were, copied nowhere
        result = KimiK3KdaPrefillCheckpointReference().reference(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, a_log=a_log, g_bias=g_bias, state_cache=state_cache,
            state_indices=state_indices, has_initial_state=has_initial_state, cu_seqlens=cu_seqlens,
            checkpoint_chunks=checkpoint_chunks, checkpoint_state=checkpoint_state, scale=scale, lower_bound=lower_bound,
        )  # fmt: skip
        state_cache.copy_(result["state_cache"])
        checkpoint_state.copy_(result["checkpoint_state"])
        return out.copy_(result["out"])

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
        ([0, 0], [NULL, NULL]),
    ],
    ids=["first", "second", "both", "no_row", "not_a_chunk", "none"],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_call_with_snapshots_takes_them_into_the_cache_rows(
    checkpoint_op_runs_reference, offsets, rows, device
):
    """On the GPU the route's plan and snapshot copy are its FlyDSL kernels; on the CPU, their torch twins."""
    # the two sequences: 5 and 3 tokens in vllm_call; long enough here for a 64-token chunk
    global SEQLENS
    saved, SEQLENS = SEQLENS, (70, 66)
    try:
        call, pages = checkpoint_call(offsets, rows)
    finally:
        SEQLENS = saved
    want_out, want_states = expected_with_snapshots(call)
    if device != "cpu":  # the cache and the snapshot rows one view of the pages there too
        pages = pages.to(device)
        cache = pages[:, : HEADS * DIM * DIM].view(CK_SLOTS, HEADS, DIM, DIM)
        call = {n: v.to(device) if torch.is_tensor(v) else v for n, v in call.items()}
        call.update(state_cache=cache, checkpoint_state=cache)
    conv_before = pages[:, HEADS * DIM * DIM :].clone()
    out, final_state = kda_api._chunk_kda_prefill(**call)
    assert final_state is None
    torch.testing.assert_close(out[0].float().cpu(), want_out, atol=2e-3, rtol=2e-2)
    states = call["state_cache"].cpu()
    written = {2, 3} | {r for o, r in zip(offsets, rows) if r != NULL and o % 64 == 0 and o > 0}
    for slot in written:
        torch.testing.assert_close(states[slot], want_states[slot], atol=1e-4, rtol=1e-4)
    # vLLM writes no row for a snapshot it does not take, its null block's (row 0) neither
    untouched = set(range(CK_SLOTS)) - written
    for slot in untouched:
        assert torch.equal(states[slot], want_states[slot])
    assert torch.equal(pages[:, HEADS * DIM * DIM :], conv_before)


@pytest.mark.gpu
@pytest.mark.parametrize("offsets_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_the_plan_and_the_copy_on_the_gpu_are_their_torch_twins(offsets_dtype, indices_dtype):
    """Past one workgroup each: 600 sequences of either integer width, and snapshots of three blocks (the last not
    whole) into cache rows a page apart."""
    from aklr.plugins.vllm.models.kimi_k3.kda_prefill._plan import checkpoint_plan, scatter_rows

    rng = torch.Generator().manual_seed(0)
    n = 600
    offsets = torch.randint(0, 4, (n,), generator=rng) * 64 + torch.randint(0, 2, (n,), generator=rng) * 3
    indices = torch.randint(
        -1, 9, (n,), generator=rng
    )  # the null block's row (0) and negative rows among them
    offsets, indices = offsets.to(offsets_dtype), indices.to(indices_dtype)
    want = checkpoint_plan(offsets, indices, 0, 64)
    got = checkpoint_plan(offsets.cuda(), indices.cuda(), 0, 64)
    assert 0 < int((want[1] >= 0).sum()) < n
    for w, g in zip(want, got, strict=True):
        assert g.dtype == w.dtype and torch.equal(g.cpu(), w)

    src = torch.randn(5, 3, 40, 80, generator=rng)
    rows = torch.tensor([7, -1, 2, -1, 0])
    pages = torch.randn(9, src[0].numel() + 16, generator=rng)
    want_pages = pages.clone()
    scatter_rows(src, rows, want_pages[:, : src[0].numel()].view(9, 3, 40, 80))
    pages = pages.cuda()
    scatter_rows(src.cuda(), rows.cuda(), pages[:, : src[0].numel()].view(9, 3, 40, 80))
    assert torch.equal(pages.cpu(), want_pages)
    assert torch.equal(want_pages[7, :-16], src[0].flatten()) and torch.equal(
        want_pages[0, :-16], src[4].flatten()
    )


def test_a_partial_checkpoint_stays_with_vllm(monkeypatch):
    monkeypatch.setattr(kda_api, "kimi_k3_kda_prefill_checkpoint", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = checkpoint_call([64, 0], [1, NULL])
    call["checkpoint_state_indices"] = None
    assert kda_api._chunk_kda_prefill(**call) is FallbackReason.OUT_OF_CONTRACT


def test_a_drawn_snapshot_point_is_inside_its_sequence():
    """What validation draws for a committed snapshot workload of several sequences is in the contract: every
    sequence at least n_tokens // n_seqs long, each snapshot point short of that."""
    contract = OpCase.of("kimi_k3_kda_prefill_checkpoint@1.0").contract
    several = 0
    for path in committed_schema_paths(op_dir(contract.op_package)):
        data = json.loads(path.read_text())
        if data["contract_id"] != contract.contract_id:
            continue
        for workload in data["workloads"]:
            if workload["axes"]["n_seqs"] == 1:
                continue
            several += 1
            case = OpCase(contract.point_from_json(data, workload["axes"]))
            for i in range(8):
                inputs = case.inputs(f"numeric/{i}")
                lengths = inputs["cu_seqlens"].diff()
                assert int(lengths.min()) >= workload["axes"]["n_tokens"] // workload["axes"]["n_seqs"]
                assert bool((inputs["checkpoint_chunks"].long() * 64 < lengths).all())
    assert several


# ---------- the baselines: vLLM's own call on the fused backend ----------


class _FakeVllm:
    """vLLM's chunk_kda_prefill and the check of its fused kernels, as modules the baselines import: records each
    call's arguments and returns vLLM's output (out when given, else v), the final state None."""

    def __init__(self, monkeypatch, fused=True):
        self.calls = []

        def chunk_kda_prefill(**kwargs):
            self.calls.append(kwargs)
            return (kwargs["out"] if kwargs["out"] is not None else kwargs["v"]), None

        modules = {
            "vllm.models.kimi_k3.amd.ops.kda_prefill": {"chunk_kda_prefill": chunk_kda_prefill},
            "vllm.models.kimi_k3.amd.ops.kda_chunk": {"can_use_fused_kda_chunk": lambda *a: fused},
            "vllm.third_party.flash_linear_attention.ops.utils": {"FLA_CHUNK_SIZE": 64},
        }
        for name, attrs in modules.items():
            monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))


@pytest.mark.parametrize("contract_id", ["kimi_k3_kda_prefill@1.0", "kimi_k3_kda_prefill_checkpoint@1.0"])
def test_the_baseline_is_vllms_fused_prefill_on_the_paged_state(monkeypatch, contract_id):
    """Each prefill baseline calls vLLM's chunk_kda_prefill as kda.py does on the fused backend: q, k, v, the gate and
    beta with vLLM's batch dim, the gate bias flat, the paged cache itself as (slot, head, v, k) with the sequences'
    rows and flags; the snapshot one, its staging rows one a sequence and the offsets in tokens."""
    vllm = _FakeVllm(monkeypatch)
    case = OpCase.of(contract_id)
    call = case.call(case.inputs())
    result = case.contract.baseline.baseline(**call)
    (got,) = vllm.calls
    assert got["backend"] == "fused" and got["use_qk_l2norm_in_kernel"] is True
    assert "initial_state" not in got and "output_final_state" not in got and "chunk_indices" not in got
    for name in ("q", "k", "v", "raw_g", "raw_beta"):
        assert torch.equal(got[name][0], call[name]) and got[name].shape[0] == 1
    heads, dim = call["q"].shape[1:]
    assert got["A_log"] is call["a_log"] and torch.equal(got["g_bias"], call["g_bias"].flatten())
    cache = got["state_cache"]
    assert cache.shape == (call["state_cache"].shape[0], heads, dim, dim)
    assert cache.data_ptr() == call["state_cache"].data_ptr() and cache.stride(0) == call[
        "state_cache"
    ].stride(0)
    assert (
        got["state_indices"] is call["state_indices"]
        and got["has_initial_state"] is call["has_initial_state"]
    )
    assert got["cu_seqlens"] is call["cu_seqlens"] and got["out"] is None
    assert (got["scale"], got["lower_bound"]) == (call["scale"], call["lower_bound"])
    if contract_id == "kimi_k3_kda_prefill_checkpoint@1.0":
        snapshots = got["checkpoint_state"]
        assert snapshots.shape == (call["checkpoint_state"].shape[0], heads, dim, dim)
        assert snapshots.data_ptr() == call["checkpoint_state"].data_ptr()
        assert torch.equal(got["checkpoint_offsets"], call["checkpoint_chunks"] * 64)
        assert "checkpoint_state_indices" not in got
    else:
        assert "checkpoint_state" not in got and "checkpoint_offsets" not in got
    assert result["out"].data_ptr() == call["v"].data_ptr()  # vLLM's output in v, without an out


def test_the_baseline_refuses_to_time_another_path(monkeypatch):
    """Without vLLM's fused kernels chunk_kda_prefill would run its Triton path: the baseline raises instead."""
    vllm = _FakeVllm(monkeypatch, fused=False)
    case = OpCase.of("kimi_k3_kda_prefill@1.0")
    with pytest.raises(RuntimeError, match="fused KDA chunk kernels"):
        case.contract.baseline.baseline(**case.call(case.inputs()))
    assert vllm.calls == []
